import cocoindex
from cocoindex import DataScope, FlowBuilder


@cocoindex.flow_def(name="EmbedMarkdownToPostgres")
def embed_markdown_to_postgres(
    flow_builder: FlowBuilder,
    data_scope: DataScope,
) -> None:
    # Add a data source that reads files from a directory
    data_scope["documents"] = flow_builder.add_source(
        cocoindex.sources.LocalFile(path="data"),
    )

    # Add a collector for data that will be exported to the vector index
    document_embeddings = data_scope.add_collector()

    with data_scope["documents"].row() as document:
        document["chunks"] = document["content"].transform(
            cocoindex.functions.SplitRecursively(),
            language="markdown",
            chunk_size=2000,
            chunk_overlap=500,
        )

        with document["chunks"].row() as chunk:
            chunk["embedding"] = chunk["text"].transform(
                cocoindex.functions.SentenceTransformerEmbed(
                    model="sentence-transformers/all-MiniLM-L6-v2",
                ),
            )

            document_embeddings.collect(
                filename=document["filename"],
                location=chunk["location"],
                text=chunk["text"],
                embedding=chunk["embedding"],
            )

    # Export the collected data to a vector index
    document_embeddings.export(
        "document_embeddings",
        cocoindex.storages.Postgres(),
        primary_key_fields=["filename", "location"],
        vector_indexes=[
            cocoindex.VectorIndexDef(
                field_name="embedding",
                metric=cocoindex.VectorSimilarityMetric.COSINE_SIMILARITY,
            ),
        ],
    )
