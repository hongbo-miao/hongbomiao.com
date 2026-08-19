package com.hongbomiao.lance;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import org.apache.flink.configuration.ConfigOption;
import org.apache.flink.configuration.ConfigOptions;
import org.apache.flink.table.connector.source.DynamicTableSource;
import org.apache.flink.table.factories.DynamicTableSourceFactory;
import org.apache.flink.table.factories.FactoryUtil;

/**
 * Registers the {@code lance} connector so a Lance dataset on object storage can be declared as a
 * Flink table and queried with the {@code VECTOR_SEARCH} table-valued function.
 *
 * <p>Discovered via the {@code META-INF/services/org.apache.flink.table.factories.Factory} SPI
 * entry, the same mechanism every built-in Flink connector (kafka, jdbc, ...) uses.
 */
public class LanceVectorSearchTableFactory implements DynamicTableSourceFactory {

  public static final ConfigOption<String> DATASET_URI =
      ConfigOptions.key("dataset-uri")
          .stringType()
          .noDefaultValue()
          .withDescription("URI of the Lance dataset, for example s3://vector-store/knowledge-base.lance.");

  public static final ConfigOption<String> DISTANCE_TYPE =
      ConfigOptions.key("distance-type")
          .stringType()
          .defaultValue("cosine")
          .withDescription("Lance distance metric: l2, cosine, or dot.");

  public static final ConfigOption<String> S3_ENDPOINT =
      ConfigOptions.key("s3.endpoint").stringType().noDefaultValue();

  public static final ConfigOption<String> S3_ACCESS_KEY =
      ConfigOptions.key("s3.access-key").stringType().noDefaultValue();

  public static final ConfigOption<String> S3_SECRET_KEY =
      ConfigOptions.key("s3.secret-key").stringType().noDefaultValue();

  public static final ConfigOption<Boolean> S3_PATH_STYLE_ACCESS =
      ConfigOptions.key("s3.path-style-access").booleanType().defaultValue(true);

  public static final ConfigOption<String> S3_REGION =
      ConfigOptions.key("s3.region").stringType().defaultValue("us-west-2");

  @Override
  public String factoryIdentifier() {
    return "lance";
  }

  @Override
  public Set<ConfigOption<?>> requiredOptions() {
    Set<ConfigOption<?>> options = new HashSet<>();
    options.add(DATASET_URI);
    return options;
  }

  @Override
  public Set<ConfigOption<?>> optionalOptions() {
    Set<ConfigOption<?>> options = new HashSet<>();
    options.add(DISTANCE_TYPE);
    options.add(S3_ENDPOINT);
    options.add(S3_ACCESS_KEY);
    options.add(S3_SECRET_KEY);
    options.add(S3_PATH_STYLE_ACCESS);
    options.add(S3_REGION);
    return options;
  }

  @Override
  public DynamicTableSource createDynamicTableSource(Context context) {
    FactoryUtil.TableFactoryHelper helper = FactoryUtil.createTableFactoryHelper(this, context);
    helper.validate();
    Map<String, String> options = helper.getOptions().toMap();

    Map<String, String> storageOptions = new HashMap<>();
    if (helper.getOptions().getOptional(S3_ENDPOINT).isPresent()) {
      storageOptions.put("aws_endpoint", options.get(S3_ENDPOINT.key()));
      storageOptions.put("aws_access_key_id", options.getOrDefault(S3_ACCESS_KEY.key(), ""));
      storageOptions.put("aws_secret_access_key", options.getOrDefault(S3_SECRET_KEY.key(), ""));
      storageOptions.put(
          "aws_virtual_hosted_style_request",
          String.valueOf(!helper.getOptions().get(S3_PATH_STYLE_ACCESS)));
      storageOptions.put("aws_region", helper.getOptions().get(S3_REGION));
      storageOptions.put("allow_http", "true");
    }

    return new LanceVectorSearchTableSource(
        helper.getOptions().get(DATASET_URI),
        helper.getOptions().get(DISTANCE_TYPE),
        storageOptions,
        context.getPhysicalRowDataType());
  }
}
