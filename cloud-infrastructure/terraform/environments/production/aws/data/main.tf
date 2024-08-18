data "terraform_remote_state" "production_aws_network_terraform_remote_state" {
  backend = "s3"
  config = {
    region = "us-west-2"
    bucket = "hm-terraform-bucket"
    key    = "production/aws/network/terraform.tfstate"
  }
}
data "aws_region" "current" {}

# Amazon S3 bucket - hm-production-bucket
module "hm_production_bucket" {
  providers      = { aws = aws.production }
  source         = "../../../../modules/aws/hm_amazon_s3_bucket"
  s3_bucket_name = "hm-production-bucket"
  environment    = var.environment
  team           = var.team
}

# Kafka KMS key
module "kafka_kms_key" {
  source           = "../../../../modules/aws/hm_aws_kms_key"
  aws_kms_key_name = "hm/kafka-kms-key"
  environment      = var.environment
  team             = var.team
}

# IoT data
locals {
  iot_data_name = "${var.environment}-iot-data"
}
# IoT data - S3 bucket
module "s3_bucket_iot_data" {
  providers      = { aws = aws.production }
  source         = "../../../../modules/aws/hm_amazon_s3_bucket"
  s3_bucket_name = "${local.iot_data_name}-bucket"
  environment    = var.environment
  team           = var.team
}
# IoT Kafka
locals {
  iot_kafka_name                    = "${var.environment}-iot-kafka"
  iot_amazon_vpc_private_subnet_ids = slice(data.terraform_remote_state.production_aws_network_terraform_remote_state.outputs.hm_amazon_vpc_private_subnets_ids, 0, 3)
}
# IoT Kafka - S3 bucket
module "s3_bucket_iot_kafka" {
  providers      = { aws = aws.production }
  source         = "../../../../modules/aws/hm_amazon_s3_bucket"
  s3_bucket_name = "${local.iot_kafka_name}-bucket"
  environment    = var.environment
  team           = var.team
}
# IoT Kafka - security group
module "iot_kafka_security_group" {
  providers                      = { aws = aws.production }
  source                         = "../../../../modules/aws/hm_amazon_msk_security_group"
  amazon_ec2_security_group_name = "${local.iot_kafka_name}-security-group"
  amazon_vpc_id                  = data.terraform_remote_state.production_aws_network_terraform_remote_state.outputs.hm_amazon_vpc_id
  amazon_vpc_cidr_ipv4           = data.terraform_remote_state.production_aws_network_terraform_remote_state.outputs.hm_amazon_vpc_id.cidr_block
  environment                    = var.environment
  team                           = var.team
}
# IoT Kafka - Kafka cluster
module "iot_kafka_cluster" {
  providers                       = { aws = aws.production }
  source                          = "../../../../modules/aws/hm_amazon_msk_cluster"
  amazon_msk_cluster_name         = local.iot_kafka_name
  kafka_version                   = "3.7.x.kraft"
  kafka_broker_instance_type      = "kafka.m7g.large"
  kafka_broker_number             = 60
  kafka_broker_log_s3_bucket_name = module.s3_bucket_iot_kafka.name
  amazon_vpc_security_group_id    = module.iot_kafka_security_group.id
  amazon_vpc_subnet_ids           = local.iot_amazon_vpc_private_subnet_ids
  amazon_ebs_volume_size_gb       = 10240
  aws_kms_key_arn                 = module.kafka_kms_key.arn
  is_scram_enabled                = true
  environment                     = var.environment
  team                            = var.team
}
# IoT Kafka - SASL/SCRAM
data "aws_secretsmanager_secret" "iot_kafka_producer_secret" {
  # https://docs.aws.amazon.com/msk/latest/developerguide/msk-password.html
  # Secret name must begin with "AmazonMSK_"
  name = "AmazonMSK_hm/production-iot-kafka/producer"
}
module "iot_kafka_sasl_scram_secret_association" {
  providers                      = { aws = aws.production }
  source                         = "../../../../modules/aws/hm_amazon_msk_cluster_sasl_scram_secret_association"
  amazon_msk_cluster_arn         = module.iot_kafka_cluster.arn
  aws_secrets_manager_secret_arn = data.aws_secretsmanager_secret.iot_kafka_producer_secret.arn
}
# IoT Kafka - S3 JSON sink plugin
locals {
  iot_kafka_s3_json_sink_plugin_name      = "${local.iot_kafka_name}-s3-json-sink-plugin"
  iot_kafka_s3_json_sink_plugin_file_name = "${local.iot_kafka_s3_json_sink_plugin_name}.zip"
  iot_kafka_s3_json_sink_plugin_dir_path  = "files/amazon-msk/${local.iot_kafka_name}/plugins/${local.iot_kafka_s3_json_sink_plugin_name}"
}
data "external" "local_iot_kafka_s3_json_sink_plugin" {
  program = ["bash", "${local.iot_kafka_s3_json_sink_plugin_dir_path}/build.sh"]
  query = {
    kafka_plugin_name                            = local.iot_kafka_s3_json_sink_plugin_name
    confluent_kafka_connect_s3_converter_version = "10.5.13" # https://www.confluent.io/hub/confluentinc/kafka-connect-s3
    local_dir_path                               = local.iot_kafka_s3_json_sink_plugin_dir_path
    local_file_name                              = local.iot_kafka_s3_json_sink_plugin_file_name
  }
}
module "s3_object_iot_kafka_s3_json_sink_plugin" {
  providers       = { aws = aws.production }
  source          = "../../../../modules/aws/hm_amazon_s3_object"
  s3_bucket_name  = module.s3_bucket_iot_kafka.name
  s3_key          = "plugins/${local.iot_kafka_s3_json_sink_plugin_file_name}"
  local_file_path = data.external.local_iot_kafka_s3_json_sink_plugin.result.local_file_path
}
module "iot_kafka_s3_json_sink_plugin" {
  providers                = { aws = aws.production }
  source                   = "../../../../modules/aws/hm_amazon_msk_plugin"
  amazon_msk_plugin_name   = local.iot_kafka_s3_json_sink_plugin_name
  s3_bucket_arn            = module.s3_bucket_iot_kafka.arn
  amazon_msk_plugin_s3_key = module.s3_object_iot_kafka_s3_json_sink_plugin.s3_key
  depends_on = [
    module.s3_object_iot_kafka_s3_json_sink_plugin
  ]
}
# IoT Kafka - S3 JSON sink connector
locals {
  iot_kafka_s3_json_sink_connector_name = "ProductionIoTKafkaS3JSONSinkConnector"
}
module "iot_kafka_s3_json_sink_connector_iam_role" {
  providers                 = { aws = aws.production }
  source                    = "../../../../modules/aws/hm_amazon_msk_s3_sink_connector_iam_role"
  amazon_msk_connector_name = local.iot_kafka_s3_json_sink_connector_name
  amazon_msk_arn            = module.iot_kafka_cluster.arn
  msk_plugin_s3_bucket_name = module.s3_bucket_iot_kafka.name
  msk_log_s3_bucket_name    = module.s3_bucket_iot_kafka.name
  msk_data_s3_bucket_name   = module.s3_bucket_iot_data.name
  environment               = var.environment
  team                      = var.team
}
module "iot_kafka_s3_json_sink_connector" {
  providers                            = { aws = aws.production }
  source                               = "../../../../modules/aws/hm_amazon_msk_s3_json_sink_connector"
  amazon_msk_connector_name            = local.iot_kafka_s3_json_sink_connector_name
  kafka_topics                         = ["production.iot.device.json"]
  aws_region                           = data.aws_region.current.name
  s3_bucket_name                       = module.s3_bucket_iot_data.name
  max_task_number                      = 3
  max_worker_number                    = 10
  worker_microcontroller_unit_number   = 1
  kafka_connect_version                = "2.7.1"
  amazon_msk_plugin_arn                = module.iot_kafka_s3_json_sink_plugin.arn
  amazon_msk_plugin_revision           = module.iot_kafka_s3_json_sink_plugin.latest_revision
  amazon_msk_connector_iam_role_arn    = module.iot_kafka_s3_json_sink_connector_iam_role.arn
  amazon_msk_cluster_bootstrap_servers = module.iot_kafka_cluster.bootstrap_servers
  amazon_vpc_security_group_id         = module.iot_kafka_security_group.id
  amazon_vpc_subnet_ids                = local.iot_amazon_vpc_private_subnet_ids
  msk_log_s3_bucket_name               = module.s3_bucket_iot_kafka.name
  msk_log_s3_key                       = "connectors/${local.iot_kafka_s3_json_sink_connector_name}"
  environment                          = var.environment
  team                                 = var.team
  depends_on = [
    module.iot_kafka_s3_json_sink_plugin
  ]
}
# IoT Kafka - S3 parquet sink plugin
locals {
  iot_kafka_s3_parquet_sink_plugin_name      = "${local.iot_kafka_name}-s3-parquet-sink-plugin"
  iot_kafka_s3_parquet_sink_plugin_file_name = "${local.iot_kafka_s3_parquet_sink_plugin_name}.zip"
  iot_kafka_s3_parquet_sink_plugin_dir_path  = "files/amazon-msk/${local.iot_kafka_name}/plugins/${local.iot_kafka_s3_parquet_sink_plugin_name}"
}
data "external" "local_iot_kafka_s3_parquet_sink_plugin" {
  program = ["bash", "${local.iot_kafka_s3_parquet_sink_plugin_dir_path}/build.sh"]
  query = {
    kafka_plugin_name                              = local.iot_kafka_s3_parquet_sink_plugin_name
    confluent_kafka_connect_s3_converter_version   = "10.5.13" # https://www.confluent.io/hub/confluentinc/kafka-connect-s3
    confluent_kafka_connect_avro_converter_version = "7.6.1"   # https://www.confluent.io/hub/confluentinc/kafka-connect-avro-converter
    local_dir_path                                 = local.iot_kafka_s3_parquet_sink_plugin_dir_path
    local_file_name                                = local.iot_kafka_s3_parquet_sink_plugin_file_name
  }
}
module "s3_object_iot_kafka_s3_parquet_sink_plugin" {
  providers       = { aws = aws.production }
  source          = "../../../../modules/aws/hm_amazon_s3_object"
  s3_bucket_name  = module.s3_bucket_iot_kafka.name
  s3_key          = "plugins/${local.iot_kafka_s3_parquet_sink_plugin_file_name}"
  local_file_path = data.external.local_iot_kafka_s3_parquet_sink_plugin.result.local_file_path
}
module "iot_kafka_s3_parquet_sink_plugin" {
  providers                = { aws = aws.production }
  source                   = "../../../../modules/aws/hm_amazon_msk_plugin"
  amazon_msk_plugin_name   = local.iot_kafka_s3_parquet_sink_plugin_name
  s3_bucket_arn            = module.s3_bucket_iot_kafka.arn
  amazon_msk_plugin_s3_key = module.s3_object_iot_kafka_s3_parquet_sink_plugin.s3_key
  depends_on = [
    module.s3_object_iot_kafka_s3_parquet_sink_plugin
  ]
}
# IoT Kafka - S3 parquet sink connector
locals {
  iot_kafka_s3_parquet_sink_connector_name = "ProductionIoTKafkaS3ParquetSinkConnector"
}
module "iot_kafka_s3_parquet_sink_connector_iam_role" {
  providers                 = { aws = aws.production }
  source                    = "../../../../modules/aws/hm_amazon_msk_s3_sink_connector_iam_role"
  amazon_msk_connector_name = local.iot_kafka_s3_parquet_sink_connector_name
  amazon_msk_arn            = module.iot_kafka_cluster.arn
  msk_plugin_s3_bucket_name = module.s3_bucket_iot_kafka.name
  msk_log_s3_bucket_name    = module.s3_bucket_iot_kafka.name
  msk_data_s3_bucket_name   = module.s3_bucket_iot_data.name
  environment               = var.environment
  team                      = var.team
}
module "iot_kafka_s3_parquet_sink_connector" {
  providers                            = { aws = aws.production }
  source                               = "../../../../modules/aws/hm_amazon_msk_s3_parquet_sink_connector"
  amazon_msk_connector_name            = local.iot_kafka_s3_parquet_sink_connector_name
  kafka_topics                         = ["production.iot.device.avro"]
  confluent_schema_registry_url        = "https://confluent-schema-registry.internal.hongbomiao.com"
  aws_region                           = data.aws_region.current.name
  s3_bucket_name                       = module.s3_bucket_iot_data.name
  max_task_number                      = 3
  max_worker_number                    = 10
  worker_microcontroller_unit_number   = 1
  kafka_connect_version                = "2.7.1"
  amazon_msk_plugin_arn                = module.iot_kafka_s3_parquet_sink_plugin.arn
  amazon_msk_plugin_revision           = module.iot_kafka_s3_parquet_sink_plugin.latest_revision
  amazon_msk_connector_iam_role_arn    = module.iot_kafka_s3_parquet_sink_connector_iam_role.arn
  amazon_msk_cluster_bootstrap_servers = module.iot_kafka_cluster.bootstrap_servers
  amazon_vpc_security_group_id         = module.iot_kafka_security_group.id
  amazon_vpc_subnet_ids                = local.iot_amazon_vpc_private_subnet_ids
  msk_log_s3_bucket_name               = module.s3_bucket_iot_kafka.name
  msk_log_s3_key                       = "connectors/${local.iot_kafka_s3_parquet_sink_connector_name}"
  environment                          = var.environment
  team                                 = var.team
  depends_on = [
    module.iot_kafka_s3_parquet_sink_plugin
  ]
}

# Tracker Kafka
locals {
  tracker_kafka_name                    = "${var.environment}-tracker-kakfa"
  tracker_amazon_vpc_private_subnet_ids = slice(data.terraform_remote_state.production_aws_network_terraform_remote_state.outputs.hm_amazon_vpc_private_subnets_ids, 0, 3)
}
# Tracker Kafka - S3 bucket
module "tracker_kafka_s3_bucket" {
  providers      = { aws = aws.production }
  source         = "../../../../modules/aws/hm_amazon_s3_bucket"
  s3_bucket_name = "${local.tracker_kafka_name}-bucket"
  environment    = var.environment
  team           = var.team
}
# Tracker Kafka - security group
module "tracker_kafka_security_group" {
  providers                      = { aws = aws.production }
  source                         = "../../../../modules/aws/hm_amazon_msk_security_group"
  amazon_ec2_security_group_name = "${local.tracker_kafka_name}-security-group"
  amazon_vpc_id                  = data.terraform_remote_state.production_aws_network_terraform_remote_state.outputs.hm_amazon_vpc_id
  amazon_vpc_cidr_ipv4           = data.terraform_remote_state.production_aws_network_terraform_remote_state.outputs.hm_amazon_vpc_id.cidr_block
  environment                    = var.environment
  team                           = var.team
}
# Tracker Kafka - Kafka cluster
module "tracker_kafka_cluster" {
  providers                       = { aws = aws.production }
  source                          = "../../../../modules/aws/hm_amazon_msk_cluster"
  amazon_msk_cluster_name         = local.tracker_kafka_name
  kafka_version                   = "3.7.x.kraft"
  kafka_broker_instance_type      = "kafka.m7g.large"
  kafka_broker_number             = 3
  kafka_broker_log_s3_bucket_name = module.tracker_kafka_s3_bucket.name
  amazon_vpc_security_group_id    = module.tracker_kafka_security_group.id
  amazon_vpc_subnet_ids           = slice(data.terraform_remote_state.production_aws_network_terraform_remote_state.outputs.hm_amazon_vpc_private_subnets_ids, 0, 3)
  amazon_ebs_volume_size_gb       = 16
  aws_kms_key_arn                 = module.kafka_kms_key.arn
  environment                     = var.environment
  team                            = var.team
}
# Tracker Kafka - Snowflake sink plugin
locals {
  tracker_kafka_snowflake_sink_plugin_name      = "${local.tracker_kafka_name}-snowflake-sink-plugin"
  tracker_kafka_snowflake_sink_plugin_file_name = "${local.tracker_kafka_snowflake_sink_plugin_name}.zip"
  tracker_kafka_snowflake_sink_plugin_dir_path  = "files/amazon-msk/${local.tracker_kafka_name}/plugins/${local.tracker_kafka_snowflake_sink_plugin_name}"
}
data "external" "local_tracker_kafka_snowflake_sink_plugin" {
  provider = aws.production
  program  = ["bash", "${local.tracker_kafka_snowflake_sink_plugin_dir_path}/build.sh"]
  query = {
    kafka_plugin_name = local.tracker_kafka_snowflake_sink_plugin_name
    # snowflake-kafka-connector requires bc-fips and bcpkix-fips
    snowflake_kafka_connector_version              = "2.2.2"   # https://mvnrepository.com/artifact/com.snowflake/snowflake-kafka-connector
    bc_fips_version                                = "1.0.2.5" # https://mvnrepository.com/artifact/org.bouncycastle/bc-fips
    bcpkix_fips_version                            = "1.0.7"   # https://mvnrepository.com/artifact/org.bouncycastle/bcpkix-fips
    confluent_kafka_connect_avro_converter_version = "7.6.1"   # https://www.confluent.io/hub/confluentinc/kafka-connect-avro-converter
    local_dir_path                                 = local.tracker_kafka_snowflake_sink_plugin_dir_path
    local_file_name                                = local.tracker_kafka_snowflake_sink_plugin_file_name
  }
}
module "s3_object_tracker_kafka_snowflake_sink_plugin" {
  providers       = { aws = aws.production }
  source          = "../../../../modules/aws/hm_amazon_s3_object"
  s3_bucket_name  = module.tracker_kafka_s3_bucket.name
  s3_key          = "plugins/${local.tracker_kafka_snowflake_sink_plugin_file_name}"
  local_file_path = data.external.local_tracker_kafka_snowflake_sink_plugin.result.local_file_path
}
module "hm_amazon_msk_plugin_tracker_kafka_snowflake_sink_plugin" {
  providers                = { aws = aws.production }
  source                   = "../../../../modules/aws/hm_amazon_msk_plugin"
  amazon_msk_plugin_name   = local.tracker_kafka_snowflake_sink_plugin_name
  s3_bucket_arn            = module.tracker_kafka_s3_bucket.arn
  amazon_msk_plugin_s3_key = module.s3_object_tracker_kafka_snowflake_sink_plugin.s3_key
  depends_on = [
    module.s3_object_tracker_kafka_snowflake_sink_plugin
  ]
}
# Tracker Kafka - Kafka sink connector
locals {
  production_tracker_sink_connector_name = "DevelopmentTrackerSinkConnector"
}
module "tracker_kafka_snowflake_sink_connector_iam_role" {
  providers                 = { aws = aws.production }
  source                    = "../../../../modules/aws/hm_amazon_msk_snowflake_sink_connector_iam"
  amazon_msk_connector_name = local.production_tracker_sink_connector_name
  amazon_msk_arn            = module.tracker_kafka_cluster.arn
  msk_plugin_s3_bucket_name = module.tracker_kafka_s3_bucket.name
  msk_log_s3_bucket_name    = module.tracker_kafka_s3_bucket.name
  environment               = var.environment
  team                      = var.team
}
data "aws_secretsmanager_secret" "tracker_snowflake_secret" {
  provider = aws.production
  name     = "hm/snowflake/production_hm_kafka_db/product/read_write"
}
data "aws_secretsmanager_secret_version" "tracker_snowflake_secret_version" {
  provider  = aws.production
  secret_id = data.aws_secretsmanager_secret.tracker_snowflake_secret.id
}
module "tracker_kafka_snowflake_sink_connector" {
  providers                            = { aws = aws.production }
  source                               = "../../../../modules/aws/hm_amazon_msk_snowflake_sink_connector"
  amazon_msk_connector_name            = local.production_tracker_sink_connector_name
  kafka_topic                          = "production.tracker.analytic-events.avro"
  max_task_number                      = 3
  max_worker_number                    = 2
  worker_microcontroller_unit_number   = 1
  snowflake_user_name                  = jsondecode(data.aws_secretsmanager_secret_version.tracker_snowflake_secret_version.secret_string)["user_name"]
  snowflake_private_key                = jsondecode(data.aws_secretsmanager_secret_version.tracker_snowflake_secret_version.secret_string)["private_key"]
  snowflake_private_key_passphrase     = jsondecode(data.aws_secretsmanager_secret_version.tracker_snowflake_secret_version.secret_string)["private_key_passphrase"]
  snowflake_role_name                  = "HM_KAFKA_DB_PRODUCT_READ_WRITE_ROLE"
  snowflake_database_name              = "HM_KAFKA_DB"
  snowflake_schema_name                = "ENGINEERING"
  confluent_schema_registry_url        = "https://confluent-schema-registry.internal.hongbomiao.com"
  kafka_connect_version                = "2.7.1"
  amazon_msk_plugin_arn                = module.hm_amazon_msk_plugin_tracker_kafka_snowflake_sink_plugin.arn
  amazon_msk_plugin_revision           = module.hm_amazon_msk_plugin_tracker_kafka_snowflake_sink_plugin.latest_revision
  amazon_msk_connector_iam_role_arn    = module.tracker_kafka_snowflake_sink_connector_iam_role.arn
  amazon_msk_cluster_bootstrap_servers = module.tracker_kafka_cluster.bootstrap_servers
  amazon_vpc_security_group_id         = module.tracker_kafka_security_group.id
  amazon_vpc_subnet_ids                = local.tracker_amazon_vpc_private_subnet_ids
  msk_log_s3_bucket_name               = module.tracker_kafka_s3_bucket.name
  msk_log_s3_key                       = "amazon-msk/connectors/${local.production_tracker_sink_connector_name}"
  environment                          = var.environment
  team                                 = var.team
  depends_on = [
    module.hm_amazon_msk_plugin_tracker_kafka_snowflake_sink_plugin
  ]
}
