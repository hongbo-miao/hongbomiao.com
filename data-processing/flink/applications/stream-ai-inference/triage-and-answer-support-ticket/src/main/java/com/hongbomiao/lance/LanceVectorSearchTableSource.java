package com.hongbomiao.lance;

import java.util.Map;
import org.apache.flink.table.connector.source.DynamicTableSource;
import org.apache.flink.table.connector.source.VectorSearchTableSource;
import org.apache.flink.table.connector.source.search.VectorSearchFunctionProvider;
import org.apache.flink.table.types.DataType;

/**
 * A {@link VectorSearchTableSource} backed by a Lance dataset on object storage.
 *
 * <p>Flink calls {@link #getSearchRuntimeProvider(VectorSearchContext)} once per {@code
 * VECTOR_SEARCH} call site, telling us via {@link VectorSearchContext#getSearchColumns()} which
 * physical column of the Lance table the query vector should be compared against.
 */
public class LanceVectorSearchTableSource implements VectorSearchTableSource, DynamicTableSource {

  private final String datasetUri;
  private final String distanceType;
  private final Map<String, String> storageOptions;
  private final DataType physicalRowType;

  public LanceVectorSearchTableSource(
      String datasetUri,
      String distanceType,
      Map<String, String> storageOptions,
      DataType physicalRowType) {
    this.datasetUri = datasetUri;
    this.distanceType = distanceType;
    this.storageOptions = storageOptions;
    this.physicalRowType = physicalRowType;
  }

  @Override
  public VectorSearchRuntimeProvider getSearchRuntimeProvider(VectorSearchContext context) {
    int[][] searchColumns = context.getSearchColumns();
    if (searchColumns.length != 1 || searchColumns[0].length != 1) {
      throw new IllegalArgumentException(
          "The lance connector only supports searching a single top-level vector column, got: "
              + java.util.Arrays.deepToString(searchColumns));
    }
    int vectorColumnIndex = searchColumns[0][0];
    return VectorSearchFunctionProvider.of(
        new LanceVectorSearchFunction(
            datasetUri, distanceType, storageOptions, physicalRowType, vectorColumnIndex));
  }

  @Override
  public DynamicTableSource copy() {
    return new LanceVectorSearchTableSource(datasetUri, distanceType, storageOptions, physicalRowType);
  }

  @Override
  public String asSummaryString() {
    return "Lance(datasetUri=" + datasetUri + ")";
  }
}
