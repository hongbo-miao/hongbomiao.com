package com.hongbomiao;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.channels.Channels;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.BigIntVector;
import org.apache.arrow.vector.Float4Vector;
import org.apache.arrow.vector.VarCharVector;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.complex.FixedSizeListVector;
import org.apache.arrow.vector.ipc.ArrowStreamReader;
import org.apache.arrow.vector.ipc.ArrowStreamWriter;
import org.apache.arrow.vector.types.FloatingPointPrecision;
import org.apache.arrow.vector.types.pojo.ArrowType;
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.arrow.vector.types.pojo.FieldType;
import org.apache.arrow.vector.types.pojo.Schema;
import org.lance.Dataset;
import org.lance.WriteParams;

/**
 * A one-shot batch job (run as a Kubernetes {@code Job}, before the Flink job starts) that embeds
 * every knowledge-base article with Ollama and writes the result as a Lance dataset, so the
 * streaming job's {@code VECTOR_SEARCH} has something to search against.
 */
public class KnowledgeBaseLoader {

  private static String getEnv(String name, String defaultValue) {
    String value = System.getenv(name);
    return value == null || value.isEmpty() ? defaultValue : value;
  }

  public static void main(String[] args) throws Exception {
    String knowledgeBasePath = getEnv("KNOWLEDGE_BASE_PATH", "/opt/flink/knowledge-base/knowledge-base.json");
    String ollamaBaseUrl = getEnv("OLLAMA_BASE_URL", "http://ollama:11434");
    String embeddingModel = getEnv("OLLAMA_EMBEDDING_MODEL", "all-minilm:22m");
    int embeddingDimension = Integer.parseInt(getEnv("EMBEDDING_DIMENSION", "384"));
    String datasetUri = getEnv("KNOWLEDGE_BASE_DATASET_URI", "s3://vector-store/knowledge-base.lance");
    String s3Endpoint = getEnv("S3_ENDPOINT", "http://rustfs-svc:9000");
    String s3AccessKey = getEnv("S3_ACCESS_KEY", "rustfs_admin");
    String s3SecretKey = getEnv("S3_SECRET_KEY", "passw0rd");

    List<JsonNode> articles = readKnowledgeBaseArticles(knowledgeBasePath);
    HttpClient httpClient = HttpClient.newHttpClient();

    try (BufferAllocator allocator = new RootAllocator()) {
      Schema schema = buildKnowledgeBaseSchema(embeddingDimension);
      try (VectorSchemaRoot root = VectorSchemaRoot.create(schema, allocator)) {
        populateKnowledgeBaseVectors(
            root, articles, httpClient, ollamaBaseUrl, embeddingModel, embeddingDimension);

        byte[] arrowStreamBytes = serializeToArrowStream(root);
        try (ArrowStreamReader reader =
            new ArrowStreamReader(new ByteArrayInputStream(arrowStreamBytes), allocator)) {
          Map<String, String> storageOptions = new HashMap<>();
          storageOptions.put("aws_endpoint", s3Endpoint);
          storageOptions.put("aws_access_key_id", s3AccessKey);
          storageOptions.put("aws_secret_access_key", s3SecretKey);
          storageOptions.put("aws_virtual_hosted_style_request", "false");
          storageOptions.put("aws_region", "us-west-2");
          storageOptions.put("allow_http", "true");

          Dataset dataset =
              Dataset.write()
                  .allocator(allocator)
                  .reader(reader)
                  .uri(datasetUri)
                  .mode(WriteParams.WriteMode.OVERWRITE)
                  .storageOptions(storageOptions)
                  .execute();
          System.out.println(
              "Wrote " + articles.size() + " knowledge-base articles to " + datasetUri);
          dataset.close();
        }
      }
    }
  }

  private static List<JsonNode> readKnowledgeBaseArticles(String path) throws Exception {
    ObjectMapper objectMapper = new ObjectMapper();
    JsonNode root = objectMapper.readTree(Files.readString(Paths.get(path), StandardCharsets.UTF_8));
    List<JsonNode> articles = new ArrayList<>();
    root.forEach(articles::add);
    return articles;
  }

  private static Schema buildKnowledgeBaseSchema(int embeddingDimension) {
    Field embeddingElementField =
        new Field(
            "item", FieldType.nullable(new ArrowType.FloatingPoint(FloatingPointPrecision.SINGLE)), null);
    // Lance only treats a column as searchable vector data if it is a FixedSizeList (or a list of
    // them) — a plain variable-length List<Float32> is rejected at query time with "Data type is not
    // a vector (FixedSizeListArray or List<FixedSizeListArray>), but Float32".
    Field embeddingField =
        new Field(
            "embedding",
            FieldType.nullable(new ArrowType.FixedSizeList(embeddingDimension)),
            List.of(embeddingElementField));
    return new Schema(
        List.of(
            new Field("id", FieldType.nullable(new ArrowType.Int(64, true)), null),
            new Field("title", FieldType.nullable(ArrowType.Utf8.INSTANCE), null),
            new Field("body", FieldType.nullable(ArrowType.Utf8.INSTANCE), null),
            embeddingField));
  }

  private static void populateKnowledgeBaseVectors(
      VectorSchemaRoot root,
      List<JsonNode> articles,
      HttpClient httpClient,
      String ollamaBaseUrl,
      String embeddingModel,
      int embeddingDimension)
      throws Exception {
    BigIntVector idVector = (BigIntVector) root.getVector("id");
    VarCharVector titleVector = (VarCharVector) root.getVector("title");
    VarCharVector bodyVector = (VarCharVector) root.getVector("body");
    FixedSizeListVector embeddingVector = (FixedSizeListVector) root.getVector("embedding");
    Float4Vector embeddingElementVector = (Float4Vector) embeddingVector.getDataVector();

    idVector.allocateNew(articles.size());
    titleVector.allocateNew();
    bodyVector.allocateNew();
    embeddingVector.allocateNew();

    for (int rowIndex = 0; rowIndex < articles.size(); rowIndex++) {
      JsonNode article = articles.get(rowIndex);
      String title = article.get("title").asText();
      String body = article.get("body").asText();
      float[] embedding = fetchEmbedding(httpClient, ollamaBaseUrl, embeddingModel, title + ". " + body);
      if (embedding.length != embeddingDimension) {
        throw new IllegalStateException(
            "Expected "
                + embeddingModel
                + " to return "
                + embeddingDimension
                + "-dimensional embeddings, but got "
                + embedding.length
                + " — set EMBEDDING_DIMENSION to match the model.");
      }

      idVector.setSafe(rowIndex, article.get("id").asLong());
      titleVector.setSafe(rowIndex, title.getBytes(StandardCharsets.UTF_8));
      bodyVector.setSafe(rowIndex, body.getBytes(StandardCharsets.UTF_8));

      int elementStartIndex = embeddingVector.startNewValue(rowIndex);
      for (int i = 0; i < embedding.length; i++) {
        embeddingElementVector.setSafe(elementStartIndex + i, embedding[i]);
      }
    }
    embeddingElementVector.setValueCount(articles.size() * embeddingDimension);
    embeddingVector.setValueCount(articles.size());
    root.setRowCount(articles.size());
  }

  /** Calls Ollama's OpenAI-compatible {@code /v1/embeddings} endpoint for a single input string. */
  private static float[] fetchEmbedding(
      HttpClient httpClient, String ollamaBaseUrl, String embeddingModel, String input)
      throws Exception {
    ObjectMapper objectMapper = new ObjectMapper();
    Map<String, Object> requestBody = new HashMap<>();
    requestBody.put("model", embeddingModel);
    requestBody.put("input", input);

    HttpRequest request =
        HttpRequest.newBuilder()
            .uri(URI.create(ollamaBaseUrl + "/v1/embeddings"))
            .header("Content-Type", "application/json")
            .POST(
                HttpRequest.BodyPublishers.ofString(
                    objectMapper.writeValueAsString(requestBody), StandardCharsets.UTF_8))
            .build();

    HttpResponse<String> response = httpClient.send(request, HttpResponse.BodyHandlers.ofString());
    if (response.statusCode() != 200) {
      throw new IllegalStateException(
          "Ollama embeddings request failed with status " + response.statusCode() + ": " + response.body());
    }
    JsonNode embeddingNode = objectMapper.readTree(response.body()).get("data").get(0).get("embedding");
    float[] embedding = new float[embeddingNode.size()];
    for (int i = 0; i < embeddingNode.size(); i++) {
      embedding[i] = (float) embeddingNode.get(i).asDouble();
    }
    return embedding;
  }

  /** Round-trips the batch through Arrow's IPC stream format so it can hand Lance an {@link
   * org.apache.arrow.vector.ipc.ArrowReader}, which is what {@code WriteDatasetBuilder} accepts.
   */
  private static byte[] serializeToArrowStream(VectorSchemaRoot root) throws Exception {
    ByteArrayOutputStream outputStream = new ByteArrayOutputStream();
    try (ArrowStreamWriter writer = new ArrowStreamWriter(root, null, Channels.newChannel(outputStream))) {
      writer.start();
      writer.writeBatch();
      writer.end();
    }
    return outputStream.toByteArray();
  }
}
