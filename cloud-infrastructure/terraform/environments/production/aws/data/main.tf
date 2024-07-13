# Amazon S3 bucket - hm-production-bucket
module "production_hm_production_bucket_amazon_s3_bucket" {
  providers      = { aws = aws.production }
  source         = "../../../../modules/aws/hm_amazon_s3_bucket"
  s3_bucket_name = "hm-production-bucket"
  environment    = var.environment
  team           = var.team
}

# Tracker Kafka
# Tracker Kafka - S3 bucket
module "hm_amazon_s3_bucket_development_tracker_kafka" {
  providers      = { aws = aws.production }
  source         = "../../../../modules/aws/hm_amazon_s3_bucket"
  s3_bucket_name = "${var.environment}-tracker-kakfa"
  environment    = var.environment
  team           = var.team
}
# Tracker Kafka - Kafka cluster
data "aws_kms_alias" "aws_kms_kafka_key" {
  name = "alias/aws/kafka"
}
locals {
  tracker_kafka_broker_number           = 3
  tracker_amazon_vpc_private_subnet_ids = local.tracker_kafka_broker_number < 4 ? slice(var.amazon_vpc_private_subnet_ids, 0, local.tracker_kafka_broker_number) : var.amazon_vpc_private_subnet_ids
}
module "hm_amazon_msk_cluster" {
  providers                       = { aws = aws.production }
  source                          = "../../../../modules/aws/hm_amazon_msk_cluster"
  amazon_msk_cluster_name         = "${var.environment}-tracker-kafka"
  kafka_version                   = "3.7.x.kraft"
  kafka_broker_instance_type      = "kafka.m7g.large"
  kafka_broker_number             = local.tracker_kafka_broker_number
  kafka_broker_log_s3_bucket_name = module.hm_amazon_s3_bucket_development_tracker_kafka.name
  amazon_vpc_security_group_id    = "sg-xxxxxxxxxxxxxxxxx"
  amazon_vpc_subnet_ids           = local.tracker_amazon_vpc_private_subnet_ids
  aws_kms_key_arn                 = data.aws_kms_alias.aws_kms_kafka_key.target_key_arn
  environment                     = var.environment
  team                            = var.team
}
# Tracker Kafka - Kafka sink plugin
data "external" "hm_local_tracker_sink_plugin" {
  program = ["bash", "files/amazon-msk/${var.environment}-tracker-kafka/plugins/build.sh"]
  query = {
    kafka_plugin_name                              = local.tracker_kafka_sink_plugin_name
    snowflake_kafka_connector_version              = "2.2.2"   # https://mvnrepository.com/artifact/com.snowflake/snowflake-kafka-connector
    bc_fips_version                                = "1.0.2.5" # https://mvnrepository.com/artifact/org.bouncycastle/bc-fips
    bcpkix_fips_version                            = "1.0.7"   # https://mvnrepository.com/artifact/org.bouncycastle/bcpkix-fips
    confluent_kafka_connect_avro_converter_version = "7.6.1"   # https://www.confluent.io/hub/confluentinc/kafka-connect-avro-converter
    local_dir_path                                 = "files/amazon-msk/${var.environment}-tracker-kafka/plugins"
    local_file_name                                = local.tracker_kafka_sink_plugin_file_name
  }
}
locals {
  tracker_kafka_sink_plugin_name      = "${var.environment}-tracker-sink-plugin"
  tracker_kafka_sink_plugin_file_name = "${local.tracker_kafka_sink_plugin_name}.zip"
}
module "hm_amazon_s3_object_tracker_kafka_sink_plugin" {
  providers       = { aws = aws.production }
  source          = "../../../../modules/aws/hm_amazon_s3_object"
  s3_bucket_name  = module.hm_amazon_s3_bucket_development_tracker_kafka.name
  s3_key          = "plugins/${local.tracker_kafka_sink_plugin_file_name}"
  local_file_path = data.external.hm_local_tracker_sink_plugin.result.local_file_path
}
module "hm_amazon_msk_plugin_tracker_kafka_sink_plugin" {
  providers                = { aws = aws.production }
  source                   = "../../../../modules/aws/hm_amazon_msk_plugin"
  amazon_msk_plugin_name   = local.tracker_kafka_sink_plugin_name
  s3_bucket_arn            = module.hm_amazon_s3_bucket_development_tracker_kafka.arn
  amazon_msk_plugin_s3_key = module.hm_amazon_s3_object_tracker_kafka_sink_plugin.s3_key
}
# Tracker Kafka - Kafka sink connector
locals {
  production_tracker_sink_connector_name = "DevelopmentTrackerSinkConnector"
}
module "hm_amazon_msk_tracker_sink_connector_iam" {
  providers                 = { aws = aws.production }
  source                    = "../../../../modules/aws/hm_amazon_msk_connector_iam"
  amazon_msk_connector_name = local.production_tracker_sink_connector_name
  amazon_msk_arn            = module.hm_amazon_msk_cluster.arn
  msk_plugin_s3_bucket_name = module.hm_amazon_s3_bucket_development_tracker_kafka.name
  msk_log_s3_bucket_name    = module.hm_amazon_s3_bucket_development_tracker_kafka.name
  environment               = var.environment
  team                      = var.team
}
data "aws_secretsmanager_secret" "tracker_snowflake_secret" {
  name = "hm/snowflake/production_hm_kafka_db/product/read_write"
}
data "aws_secretsmanager_secret_version" "tracker_snowflake_secret_version" {
  secret_id = data.aws_secretsmanager_secret.tracker_snowflake_secret.id
}
module "hm_amazon_msk_tracker_sink_connector" {
  providers                            = { aws = aws.production }
  source                               = "../../../../modules/aws/hm_amazon_msk_connector"
  amazon_msk_connector_name            = local.production_tracker_sink_connector_name
  kafka_connect_version                = "2.7.1"
  amazon_msk_plugin_arn                = module.hm_amazon_msk_plugin_tracker_kafka_sink_plugin.arn
  amazon_msk_plugin_revision           = module.hm_amazon_msk_plugin_tracker_kafka_sink_plugin.latest_revision
  amazon_msk_connector_iam_role_arn    = module.hm_amazon_msk_tracker_sink_connector_iam.arn
  amazon_msk_cluster_bootstrap_servers = module.hm_amazon_msk_cluster.bootstrap_servers
  confluent_schema_registry_url        = "https://production-confluent-schema-registry.hongbomiao.com"
  snowflake_user_name                  = jsondecode(data.aws_secretsmanager_secret_version.tracker_snowflake_secret_version.secret_string)["user_name"]
  snowflake_private_key                = jsondecode(data.aws_secretsmanager_secret_version.tracker_snowflake_secret_version.secret_string)["private_key"]
  snowflake_private_key_passphrase     = jsondecode(data.aws_secretsmanager_secret_version.tracker_snowflake_secret_version.secret_string)["private_key_passphrase"]
  snowflake_role_name                  = "HM_DEVELOPMENT_HM_KAFKA_DB_PRODUCT_READ_WRITE_ROLE"
  msk_log_s3_bucket_name               = module.hm_amazon_s3_bucket_development_tracker_kafka.name
  msk_log_s3_key                       = "amazon-msk/connectors/${local.production_tracker_sink_connector_name}"
  kafka_topic_name                     = "production.tracker.analytic-events.avro"
  snowflake_database_name              = "DEVELOPMENT_HM_KAFKA_DB"
  snowflake_schema_name                = "ENGINEERING"
  environment                          = var.environment
  team                                 = var.team
}
