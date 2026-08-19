package com.hongbomiao.lance;

import static com.hongbomiao.lance.ConvertArrowRowToRowData.convertArrowRowToRowData;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Map;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.ipc.ArrowReader;
import org.apache.flink.table.data.GenericRowData;
import org.apache.flink.table.data.RowData;
import org.apache.flink.table.data.utils.JoinedRowData;
import org.apache.flink.table.functions.FunctionContext;
import org.apache.flink.table.functions.VectorSearchFunction;
import org.apache.flink.table.types.DataType;
import org.apache.flink.table.types.logical.RowType;
import org.lance.Dataset;
import org.lance.ReadOptions;
import org.lance.index.DistanceType;
import org.lance.ipc.LanceScanner;
import org.lance.ipc.Query;
import org.lance.ipc.ScanOptions;

/**
 * Runs a Lance approximate/exact nearest-neighbor scan for each input row.
 *
 * <p>Matches Flink's contract for {@link VectorSearchFunction} demonstrated by Flink's own {@code
 * TestValueVectorSearchFunction}: the returned rows are the matched table's full physical row
 * joined with a trailing {@code score} column, highest-similarity first.
 */
public class LanceVectorSearchFunction extends VectorSearchFunction {

  private final String datasetUri;
  private final DistanceType distanceType;
  private final Map<String, String> storageOptions;
  private final RowType physicalRowType;
  private final int vectorColumnIndex;

  private transient Dataset dataset;
  private transient String lanceVectorColumnName;

  public LanceVectorSearchFunction(
      String datasetUri,
      String distanceType,
      Map<String, String> storageOptions,
      DataType physicalRowType,
      int vectorColumnIndex) {
    this.datasetUri = datasetUri;
    this.distanceType = parseDistanceType(distanceType);
    this.storageOptions = storageOptions;
    this.physicalRowType = (RowType) physicalRowType.getLogicalType();
    this.vectorColumnIndex = vectorColumnIndex;
  }

  private static DistanceType parseDistanceType(String distanceType) {
    for (DistanceType candidate : DistanceType.values()) {
      if (candidate.name().equalsIgnoreCase(distanceType)) {
        return candidate;
      }
    }
    throw new IllegalArgumentException(
        "Unknown distance-type '"
            + distanceType
            + "', expected one of (case-insensitive): "
            + java.util.Arrays.toString(DistanceType.values()));
  }

  @Override
  public void open(FunctionContext context) throws Exception {
    super.open(context);
    ReadOptions readOptions = new ReadOptions.Builder().setStorageOptions(storageOptions).build();
    dataset = Dataset.open(datasetUri, readOptions);
    // Resolve the column to search by position rather than by the Flink column's name: a
    // VECTOR_SEARCH join places the search table's columns beside the query table's, so the Flink
    // table often has to rename them to keep the joined row's field names unique.
    lanceVectorColumnName = dataset.getSchema().getFields().get(vectorColumnIndex).getName();
  }

  @Override
  public Collection<RowData> vectorSearch(int topK, RowData queryData) throws IOException {
    float[] queryVector = queryData.getArray(0).toFloatArray();

    Query nearest =
        new Query.Builder()
            .setColumn(lanceVectorColumnName)
            .setKey(queryVector)
            .setK(topK)
            .setDistanceType(distanceType)
            .build();
    ScanOptions scanOptions = new ScanOptions.Builder().nearest(nearest).build();

    Collection<RowData> results = new ArrayList<>();
    LanceScanner scanner = dataset.newScan(scanOptions);
    try (ArrowReader reader = scanner.scanBatches()) {
      while (reader.loadNextBatch()) {
        VectorSchemaRoot batch = reader.getVectorSchemaRoot();
        for (int rowIndex = 0; rowIndex < batch.getRowCount(); rowIndex++) {
          RowData matchedRow = convertArrowRowToRowData(batch, rowIndex, physicalRowType);
          // Lance appends a "_distance" column (org.lance.ipc.Query) for nearest-neighbor
          // scans; with COSINE distance that is 1 - cosine_similarity, so similarity = 1 - distance.
          float distance = batch.getVector("_distance").getObject(rowIndex) != null
              ? ((Number) batch.getVector("_distance").getObject(rowIndex)).floatValue()
              : 0f;
          double score = 1.0 - distance;
          results.add(new JoinedRowData(matchedRow, GenericRowData.of(score)));
        }
      }
    } finally {
      try {
        scanner.close();
      } catch (Exception exception) {
        throw new IOException("Failed to close Lance scanner", exception);
      }
    }
    return results;
  }

  @Override
  public void close() throws Exception {
    if (dataset != null) {
      dataset.close();
    }
    super.close();
  }
}
