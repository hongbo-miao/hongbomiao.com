data "terraform_remote_state" "hm_terraform_remote_state_production_aws_network" {
  backend = "s3"
  config = {
    region = "us-west-2"
    bucket = "hm-terraform-bucket"
    key    = "production/aws/network/terraform.tfstate"
  }
}

# Amazon S3 bucket - hm-production-bucket
module "production_hm_production_bucket_amazon_s3_bucket" {
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

# IoT Kafka
locals {
  iot_kafka_name = "hm-${var.environment}-iot-kakfa"
}
# IoT Kafka - S3 bucket
module "iot_kafka_s3_bucket" {
  source         = "../../../../modules/aws/hm_amazon_s3_bucket"
  s3_bucket_name = "${local.iot_kafka_name}-bucket"
  environment    = var.environment
  team           = var.team
}
# IoT Kafka - security group
module "iot_kafka_security_group" {
  source                         = "../../../../modules/aws/hm_amazon_msk_security_group"
  amazon_ec2_security_group_name = "${local.iot_kafka_name}-security-group"
  amazon_vpc_id                  = data.terraform_remote_state.hm_terraform_remote_state_production_aws_network.outputs.hm_amazon_vpc_id
  amazon_vpc_cidr_ipv4           = "172.16.0.0/12"
  environment                    = var.environment
  team                           = var.team
}
# IoT Kafka - Kafka cluster
module "iot_kafka_cluster" {
  source                          = "../../../../modules/aws/hm_amazon_msk_cluster"
  amazon_msk_cluster_name         = local.iot_kafka_name
  kafka_version                   = "3.7.x.kraft"
  kafka_broker_instance_type      = "kafka.m7g.large"
  kafka_broker_number             = 60
  kafka_broker_log_s3_bucket_name = module.iot_kafka_s3_bucket.name
  amazon_vpc_security_group_id    = module.iot_kafka_security_group.id
  amazon_vpc_subnet_ids           = slice(var.amazon_vpc_private_subnet_ids, 0, 3)
  amazon_ebs_volume_size_gb       = 102400
  aws_kms_key_arn                 = module.kafka_kms_key.arn
  is_scram_enabled                = true
  environment                     = var.environment
  team                            = var.team
}
# IoT Kafka - SASL/SCRAM
data "aws_secretsmanager_secret" "iot_kafka_producer_secret" {
  name = "AmazonMSK_hm/production-iot-kafka/producer"
}
module "iot_kafka_sasl_scram_secret_association" {
  source                         = "../../../../modules/aws/hm_amazon_msk_cluster_sasl_scram_secret_association"
  amazon_msk_cluster_arn         = module.iot_kafka_cluster.arn
  aws_secrets_manager_secret_arn = data.aws_secretsmanager_secret.iot_kafka_producer_secret.arn
}

# Tracker Kafka
locals {
  tracker_kafka_name                    = "hm-${var.environment}-tracker-kakfa"
  tracker_amazon_vpc_private_subnet_ids = slice(var.amazon_vpc_private_subnet_ids, 0, 3)
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
  amazon_vpc_id                  = data.terraform_remote_state.hm_terraform_remote_state_production_aws_network.outputs.hm_amazon_vpc_id
  amazon_vpc_cidr_ipv4           = "172.16.0.0/12"
  environment                    = var.environment
  team                           = var.team
}
# Tracker Kafka - Kafka cluster
module "hm_amazon_msk_cluster" {
  providers                       = { aws = aws.production }
  source                          = "../../../../modules/aws/hm_amazon_msk_cluster"
  amazon_msk_cluster_name         = local.tracker_kafka_name
  kafka_version                   = "3.7.x.kraft"
  kafka_broker_instance_type      = "kafka.m7g.large"
  kafka_broker_number             = 3
  kafka_broker_log_s3_bucket_name = module.tracker_kafka_s3_bucket.name
  amazon_vpc_security_group_id    = module.tracker_kafka_security_group.id
  amazon_vpc_subnet_ids           = slice(var.amazon_vpc_private_subnet_ids, 0, 3)
  amazon_ebs_volume_size_gb       = 16
  aws_kms_key_arn                 = module.kafka_kms_key.arn
  environment                     = var.environment
  team                            = var.team
}
# Tracker Kafka - Kafka sink plugin
data "external" "hm_local_tracker_sink_plugin" {
  provider = aws.production
  program  = ["bash", "files/amazon-msk/${var.environment}-tracker-kafka/plugins/build.sh"]
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
  s3_bucket_name  = module.tracker_kafka_s3_bucket.name
  s3_key          = "plugins/${local.tracker_kafka_sink_plugin_file_name}"
  local_file_path = data.external.hm_local_tracker_sink_plugin.result.local_file_path
}
module "hm_amazon_msk_plugin_tracker_kafka_sink_plugin" {
  providers                = { aws = aws.production }
  source                   = "../../../../modules/aws/hm_amazon_msk_plugin"
  amazon_msk_plugin_name   = local.tracker_kafka_sink_plugin_name
  s3_bucket_arn            = module.tracker_kafka_s3_bucket.arn
  amazon_msk_plugin_s3_key = module.hm_amazon_s3_object_tracker_kafka_sink_plugin.s3_key
  depends_on = [
    module.hm_amazon_s3_object_tracker_kafka_sink_plugin
  ]
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
  snowflake_role_name                  = "HM_hmHM_KAFKA_DB_PRODUCT_READ_WRITE_ROLE"
  msk_log_s3_bucket_name               = module.tracker_kafka_s3_bucket.name
  msk_log_s3_key                       = "amazon-msk/connectors/${local.production_tracker_sink_connector_name}"
  kafka_topic_name                     = "production.tracker.analytic-events.avro"
  snowflake_database_name              = "hmHM_KAFKA_DB"
  snowflake_schema_name                = "ENGINEERING"
  environment                          = var.environment
  team                                 = var.team
  depends_on = [
    module.hm_amazon_msk_plugin_tracker_kafka_sink_plugin
  ]
}
