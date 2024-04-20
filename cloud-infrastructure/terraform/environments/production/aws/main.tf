terraform {
  backend "s3" {
    region = "us-west-2"
    bucket = "hm-terraform-bucket"
    key    = "production/aws/terraform.tfstate"
  }
  required_providers {
    # https://registry.terraform.io/providers/hashicorp/aws/latest
    aws = {
      source  = "hashicorp/aws"
      version = "5.46.0"
    }
    # https://registry.terraform.io/providers/hashicorp/awscc/latest
    awscc = {
      source  = "hashicorp/awscc"
      version = "0.74.0"
    }
  }
  # terraform version
  required_version = ">= 1.7"
}

provider "aws" {
  alias  = "production"
  region = "us-west-2"
}
provider "awscc" {
  alias  = "production"
  region = "us-west-2"
}

# Amazon EC2
module "production_hm_ec2_module" {
  providers         = { aws = aws.production }
  source            = "../../../modules/aws/hm_amazon_ec2"
  ec2_instance_name = "hm-ec2-instance"
  ec2_instance_ami  = "ami-0c79a55dda52434da" # Ubuntu Server 22.04 LTS (HVM), SSD Volume Type (64-bit (Arm))
  ec2_instance_type = "t2.nano"
}

# Amazon S3 bucket - hm-production-bucket
module "production_amazon_hm_production_s3_bucket" {
  providers      = { aws = aws.production }
  source         = "../../../modules/aws/hm_amazon_s3_bucket"
  s3_bucket_name = "hm-production-bucket"
  environment    = var.environment
  team           = var.team
}

# Amazon EMR
# Amazon EMR - Trino
data "aws_secretsmanager_secret" "hm_rds_secret" {
  name = "hm/amazon-rds/production_hm_iot_db/public/read_only"
}
data "aws_secretsmanager_secret_version" "hm_rds_secret_version" {
  secret_id = data.aws_secretsmanager_secret.hm_rds_secret.id
}
module "production_hm_trino_s3_set_up_script" {
  providers       = { aws = aws.production }
  source          = "../../../modules/aws/hm_amazon_s3_object"
  s3_bucket_name  = module.production_amazon_hm_production_s3_bucket.name
  s3_key          = "amazon-emr/hm-amazon-emr-cluster-trino/bootstrap-actions/set_up.sh"
  local_file_path = "./data/amazon-emr/hm-amazon-emr-cluster-trino/bootstrap-actions/set_up.sh"
}
module "production_hm_trino_emr" {
  providers                                  = { aws = aws.production }
  source                                     = "../../../modules/aws/hm_amazon_emr_cluster"
  amazon_emr_cluster_name                    = "hm-trino"
  amazon_emr_version                         = "emr-7.0.0"
  applications                               = ["Trino"]
  primary_instance_target_on_demand_capacity = 1
  primary_instance_weighted_capacity         = 1
  primary_instance_type                      = "r7g.xlarge"
  core_instance_target_on_demand_capacity    = 1
  core_instance_weighted_capacity            = 1
  core_instance_type                         = "r7g.xlarge"
  bootstrap_set_up_script_s3_uri             = module.production_hm_trino_s3_set_up_script.uri
  configurations_json_string                 = <<EOF
    [
      {
        "Classification": "delta-defaults",
        "Properties": {
          "delta.enabled": "true"
        }
      },
      {
        "Classification": "trino-connector-delta",
        "Properties": {
          "hive.metastore": "glue"
        }
      },
      {
        "Classification": "trino-connector-postgresql",
        "Properties": {
          "connection-url": "jdbc:postgresql://${jsondecode(data.aws_secretsmanager_secret_version.hm_rds_secret_version.secret_string)["host"]}:${jsondecode(data.aws_secretsmanager_secret_version.hm_rds_secret_version.secret_string)["port"]}/${jsondecode(data.aws_secretsmanager_secret_version.hm_rds_secret_version.secret_string)["database"]}",
          "connection-user": "${jsondecode(data.aws_secretsmanager_secret_version.hm_rds_secret_version.secret_string)["user_name"]}",
          "connection-password": "${jsondecode(data.aws_secretsmanager_secret_version.hm_rds_secret_version.secret_string)["password"]}"
        }
      }
    ]
  EOF
  iam_role_arn                               = "arn:aws:iam::272394222652:role/service-role/AmazonEMR-ServiceRole-hm"
  environment                                = var.environment
  team                                       = var.team
}
module "production_hm_trino_task_instance_fleet" {
  providers                          = { aws = aws.production }
  source                             = "../../../modules/aws/hm_amazon_emr_cluster_task_instance_fleet"
  amazon_emr_cluster_id              = module.production_hm_trino_emr.id
  task_instance_target_spot_capacity = 1
  task_instance_configs = [
    {
      instance_type     = "r7g.xlarge"
      weighted_capacity = 1
    },
    {
      instance_type     = "r6g.xlarge"
      weighted_capacity = 1
    }
  ]
}
data "aws_instance" "hm_trino_primary_node_ec2_instance" {
  filter {
    name   = "private-dns-name"
    values = [module.production_hm_trino_emr.master_public_dns]
  }
}
module "production_hm_route_53_record" {
  providers                     = { aws = aws.production }
  source                        = "../../../modules/aws/hm_amazon_route_53"
  amazon_route_53_record_name   = "hm-emr-trino"
  amazon_route_53_record_values = [data.aws_instance.hm_trino_primary_node_ec2_instance.private_ip]
}

# Amazon EMR - Apache Sedona
module "production_hm_sedona_s3_set_up_script" {
  providers       = { aws = aws.production }
  source          = "../../../modules/aws/hm_amazon_s3_object"
  s3_bucket_name  = module.production_amazon_hm_production_s3_bucket.name
  s3_key          = "amazon-emr/clusters/hm-amazon-emr-cluster-sedona/bootstrap-actions/set_up.sh"
  local_file_path = "./data/amazon-emr/hm-amazon-emr-cluster-sedona/bootstrap-actions/set_up.sh"
}
module "production_hm_sedona_s3_validate_python_version_script" {
  providers       = { aws = aws.production }
  source          = "../../../modules/aws/hm_amazon_s3_object"
  s3_bucket_name  = module.production_amazon_hm_production_s3_bucket.name
  s3_key          = "amazon-emr/clusters/hm-amazon-emr-cluster-sedona/steps/validate_python_version.py"
  local_file_path = "./data/amazon-emr/hm-amazon-emr-cluster-sedona/steps/validate_python_version.py"
}
module "production_hm_sedona_s3_set_up_jupyterlab_script" {
  providers       = { aws = aws.production }
  source          = "../../../modules/aws/hm_amazon_s3_object"
  s3_bucket_name  = module.production_amazon_hm_production_s3_bucket.name
  s3_key          = "amazon-emr/clusters/hm-amazon-emr-cluster-sedona/steps/set_up_jupyterlab.sh"
  local_file_path = "./data/amazon-emr/hm-amazon-emr-cluster-sedona/steps/set_up_jupyterlab.sh"
}
module "production_hm_sedona_emr" {
  providers                                  = { aws = aws.production }
  source                                     = "../../../modules/aws/hm_amazon_emr_cluster"
  amazon_emr_cluster_name                    = "hm-sedona"
  amazon_emr_version                         = "emr-7.0.0"
  applications                               = ["Hadoop", "Hive", "JupyterEnterpriseGateway", "Livy", "Spark"]
  primary_instance_target_on_demand_capacity = 1
  primary_instance_weighted_capacity         = 1
  primary_instance_type                      = "r7g.xlarge"
  core_instance_target_on_demand_capacity    = 1
  core_instance_weighted_capacity            = 1
  core_instance_type                         = "r7g.xlarge"
  bootstrap_set_up_script_s3_uri             = module.production_hm_sedona_s3_set_up_script.uri
  configurations_json_string                 = <<EOF
    [
      {
        "Classification" : "delta-defaults",
        "Properties" : {
          "delta.enabled" : "true"
        }
      },
      {
        "Classification": "spark",
        "Properties": {
          "maximizeResourceAllocation": "true"
        }
      },
      {
        "Classification" : "spark-hive-site",
        "Properties" : {
          "hive.metastore.client.factory.class" : "com.amazonaws.glue.catalog.metastore.AWSGlueDataCatalogHiveClientFactory"
        }
      },
      {
        "Classification" : "spark-defaults",
        "Properties" : {
          "spark.yarn.dist.jars" : "/usr/lib/spark/jars/geotools-wrapper-1.5.1-28.2.jar,/usr/lib/spark/jars/sedona-spark-shaded-3.5_2.12-1.5.1.jar",
          "spark.serializer" : "org.apache.spark.serializer.KryoSerializer",
          "spark.kryo.registrator" : "org.apache.sedona.core.serde.SedonaKryoRegistrator",
          "spark.sql.extensions" : "org.apache.sedona.viz.sql.SedonaVizExtensions,org.apache.sedona.sql.SedonaSqlExtensions"
        }
      }
    ]
  EOF
  iam_role_arn                               = "arn:aws:iam::272394222652:role/service-role/AmazonEMR-ServiceRole-hm"
  environment                                = var.environment
  team                                       = var.team
}
module "production_hm_sedona_emr_task_instance_fleet" {
  providers                          = { aws = aws.production }
  source                             = "../../../modules/aws/hm_amazon_emr_cluster_task_instance_fleet"
  amazon_emr_cluster_id              = module.production_hm_sedona_emr.id
  task_instance_target_spot_capacity = 2
  task_instance_configs = [
    {
      instance_type     = "r7g.2xlarge"
      weighted_capacity = 2
    },
    {
      instance_type     = "r6g.2xlarge"
      weighted_capacity = 2
    }
  ]
}
module "production_hm_sedona_emr_managed_scaling_policy" {
  providers             = { aws = aws.production }
  source                = "../../../modules/aws/hm_amazon_emr_managed_scaling_policy"
  amazon_emr_cluster_id = module.production_hm_sedona_emr.id
  max_capacity_units    = 60
}
data "aws_instance" "hm_sedona_emr_primary_node" {
  filter {
    name   = "private-dns-name"
    values = [module.production_hm_sedona_emr.master_public_dns]
  }
}
module "production_hm_sedona_route_53" {
  providers                     = { aws = aws.production }
  source                        = "../../../modules/aws/hm_amazon_route_53"
  amazon_route_53_record_name   = "hm-sedona"
  amazon_route_53_record_values = [data.aws_instance.hm_sedona_emr_primary_node.private_ip]
}

module "production_hm_sedona_emr_studio_iam" {
  providers              = { aws = aws.production }
  source                 = "../../../modules/aws/hm_amazon_emr_studio_iam"
  amazon_emr_studio_name = "hm-sedona-emr-studio"
  s3_bucket              = module.production_amazon_hm_production_s3_bucket.name
  environment            = var.environment
  team                   = var.team
}
module "production_hm_sedona_emr_studio" {
  providers              = { aws = aws.production }
  source                 = "../../../modules/aws/hm_amazon_emr_studio"
  amazon_emr_studio_name = "hm-sedona-emr-studio"
  s3_uri                 = "s3://hm-production-bucket/amazon-emr/studio/hm-sedona-emr-studio"
  iam_role_arn           = module.production_hm_sedona_emr_studio_iam.arn
  environment            = var.environment
  team                   = var.team
}

# AWS Glue DataBrew job
# AWS Glue DataBrew recipe job - ADS-B 2x Flight Trace - Write CSV to Parquet
module "production_hm_glue_databrew_recipe_job_write_adsb_2x_flight_trace_csv_to_parquet_iam" {
  providers                      = { aws = aws.production }
  source                         = "../../../modules/aws/hm_aws_glue_databrew_iam"
  aws_glue_databrew_job_nickname = "write-adsb-csv-to-parquet"
  input_s3_bucket_name           = module.production_amazon_hm_production_s3_bucket.name
  output_s3_bucket_name          = module.production_amazon_hm_production_s3_bucket.name
  environment                    = var.environment
  team                           = var.team
}
module "production_hm_glue_databrew_recipe_job_write_adsb_2x_flight_trace_csv_to_parquet_dataset" {
  providers                      = { awscc = awscc.production }
  source                         = "../../../modules/aws/hm_aws_glue_databrew_dataset_adsb_raw_data"
  for_each                       = toset(var.adsb_2x_flight_trace_raw_data_dates)
  aws_glue_databrew_dataset_name = "adsb-2x-flight-trace-dataset-raw-data-${replace(each.value, "/", "-")}"
  input_s3_bucket_name           = module.production_amazon_hm_production_s3_bucket.name
  input_s3_key                   = "data/raw/adsb_2x_flight_trace_data/${each.value}/"
  environment                    = var.environment
  team                           = var.team
}
module "production_hm_glue_databrew_recipe_job_write_adsb_2x_flight_trace_csv_to_parquet" {
  providers                         = { awscc = awscc.production }
  source                            = "../../../modules/aws/hm_aws_glue_databrew_recipe_job"
  for_each                          = toset(var.adsb_2x_flight_trace_raw_data_dates)
  aws_glue_databrew_recipe_job_name = "hm-write-adsb-2x-flight-trace-csv-to-parquet-${replace(each.value, "/", "-")}"
  aws_glue_databrew_dataset_name    = "adsb-2x-flight-trace-dataset-raw-data-${replace(each.value, "/", "-")}"
  recipe_name                       = "adsb-2x-flight-trace-recipe"
  recipe_version                    = "1.0"
  spark_worker_max_number           = 24
  timeout_min                       = 1440
  output_s3_bucket_name             = module.production_amazon_hm_production_s3_bucket.name
  output_s3_key                     = "data/raw-parquet/adsb_2x_flight_trace_data/${each.value}/"
  output_max_file_number            = 24
  iam_role_arn                      = module.production_hm_glue_databrew_recipe_job_write_adsb_2x_flight_trace_csv_to_parquet_iam.arn
  environment                       = var.environment
  team                              = var.team
  depends_on = [
    module.production_hm_glue_databrew_recipe_job_write_adsb_2x_flight_trace_csv_to_parquet_dataset
  ]
}

# AWS Glue DataBrew profile job - ADS-B 2x Flight Trace - Profile Parquet
module "production_hm_glue_databrew_profile_job_profile_adsb_2x_flight_trace_raw_parquet_iam" {
  providers                      = { aws = aws.production }
  source                         = "../../../modules/aws/hm_aws_glue_databrew_iam"
  aws_glue_databrew_job_nickname = "profile-adsb-raw-parquet"
  input_s3_bucket_name           = module.production_amazon_hm_production_s3_bucket.name
  output_s3_bucket_name          = module.production_amazon_hm_production_s3_bucket.name
  environment                    = var.environment
  team                           = var.team
}
module "production_hm_glue_databrew_profile_job_profile_adsb_2x_flight_trace_raw_parquet_dataset" {
  providers                      = { awscc = awscc.production }
  source                         = "../../../modules/aws/hm_aws_glue_databrew_dataset_adsb_raw_parquet"
  for_each                       = toset(var.adsb_2x_flight_trace_raw_parquet_dates)
  aws_glue_databrew_dataset_name = "adsb-2x-flight-trace-dataset-raw-parquet-${replace(each.value, "/", "-")}"
  input_s3_bucket_name           = module.production_amazon_hm_production_s3_bucket.name
  input_s3_key                   = "data/raw-parquet/adsb_2x_flight_trace_data/${each.value}/"
  environment                    = var.environment
  team                           = var.team
}
module "production_hm_glue_databrew_profile_job_profile_adsb_2x_flight_trace_raw_parquet" {
  providers                          = { awscc = awscc.production }
  source                             = "../../../modules/aws/hm_aws_glue_databrew_profile_job"
  for_each                           = toset(var.adsb_2x_flight_trace_raw_parquet_dates)
  aws_glue_databrew_profile_job_name = "hm-profile-adsb-2x-flight-trace-raw-parquet-${replace(each.value, "/", "-")}"
  aws_glue_databrew_dataset_name     = "adsb-2x-flight-trace-dataset-raw-parquet-${replace(each.value, "/", "-")}"
  spark_worker_max_number            = 24
  timeout_min                        = 1440
  output_s3_bucket_name              = module.production_amazon_hm_production_s3_bucket.name
  output_s3_key                      = "aws-glue-databrew/profile-results/"
  iam_role_arn                       = module.production_hm_glue_databrew_profile_job_profile_adsb_2x_flight_trace_raw_parquet_iam.arn
  environment                        = var.environment
  team                               = var.team
  depends_on = [
    module.production_hm_glue_databrew_profile_job_profile_adsb_2x_flight_trace_raw_parquet_dataset
  ]
}

# AWS Glue job
# AWS Glue job - ADS-B 2x Flight Trace
module "production_hm_glue_job_write_parquet_to_delta_table_adsb_2x_flight_trace_data_script_iam" {
  providers             = { aws = aws.production }
  source                = "../../../modules/aws/hm_aws_glue_iam"
  aws_glue_job_nickname = "write-adsb-parquet-to-delta-table"
  input_s3_bucket_name  = module.production_amazon_hm_production_s3_bucket.name
  output_s3_bucket_name = module.production_amazon_hm_production_s3_bucket.name
  environment           = var.environment
  team                  = var.team
}
module "production_hm_glue_job_write_parquet_to_delta_table_adsb_2x_flight_trace_data_script" {
  providers       = { aws = aws.production }
  source          = "../../../modules/aws/hm_amazon_s3_object"
  s3_bucket_name  = module.production_amazon_hm_production_s3_bucket.name
  s3_key          = "aws-glue/spark-scripts/hm_write_parquet_to_delta_table_adsb_2x_flight_trace_data.py"
  local_file_path = "./data/aws-glue/spark-scripts/src/hm_write_parquet_to_delta_table_adsb_2x_flight_trace_data.py"
}
module "production_hm_glue_job_write_parquet_to_delta_table_adsb_2x_flight_trace_data" {
  providers               = { aws = aws.production }
  source                  = "../../../modules/aws/hm_aws_glue_job"
  aws_glue_job_name       = "hm_write_parquet_to_delta_table_adsb_2x_flight_trace_data"
  spark_script_s3_uri     = module.production_hm_glue_job_write_parquet_to_delta_table_adsb_2x_flight_trace_data_script.uri
  spark_worker_type       = "G.1X"
  spark_worker_max_number = 900
  timeout_min             = 360
  iam_role_arn            = module.production_hm_glue_job_write_parquet_to_delta_table_adsb_2x_flight_trace_data_script_iam.arn
  environment             = var.environment
  team                    = var.team
}

# AWS Glue job - Motor
module "production_hm_glue_job_write_parquet_to_delta_table_motor_data_script" {
  providers       = { aws = aws.production }
  source          = "../../../modules/aws/hm_amazon_s3_object"
  s3_bucket_name  = module.production_amazon_hm_production_s3_bucket.name
  s3_key          = "aws-glue/spark-scripts/hm_write_parquet_to_delta_table_motor_data.py"
  local_file_path = "./data/aws-glue/spark-scripts/src/hm_write_parquet_to_delta_table_motor_data.py"
}
module "production_hm_glue_job_write_parquet_to_delta_table_motor_data" {
  providers               = { aws = aws.production }
  source                  = "../../../modules/aws/hm_aws_glue_job"
  aws_glue_job_name       = "hm_write_parquet_to_delta_lake_motor_data"
  spark_script_s3_uri     = "s3://hm-production-bucket/aws-glue/spark-scripts/hm_write_parquet_to_delta_lake_motor_data.py"
  spark_worker_type       = "G.1X"
  spark_worker_max_number = 20
  timeout_min             = 360
  iam_role_arn            = "arn:aws:iam::272394222652:role/service-role/AWSGlueServiceRole-hm"
  environment             = var.environment
  team                    = var.team
}
module "production_hm_glue_crawler_motor_data" {
  providers                     = { aws = aws.production }
  source                        = "../../../modules/aws/hm_aws_glue_crawler"
  aws_glue_crawler_name         = "hm-delta-lake-crawler-iot"
  aws_glue_crawler_delta_tables = ["s3://hm-production-bucket/delta-tables/motor_data/"]
  aws_glue_database             = "production_hm_delta_db"
  iam_role_arn                  = "arn:aws:iam::272394222652:role/service-role/AWSGlueServiceRole-hm"
  environment                   = var.environment
  team                          = var.team
}

# Amazon MSK
module "production_hm_amazon_msk_cluster" {
  providers               = { aws = aws.production }
  source                  = "../../../modules/aws/hm_amazon_msk_cluster"
  amazon_msk_cluster_name = "tracker-kafka"
  kafka_version           = "3.6.0"
  instance_type           = "kafka.m7g.large"
  kafka_broker_number     = 2
  environment             = var.environment
  team                    = var.team
}
locals {
  tracker_sink_plugin_file_name = "tracker-sink-plugin.zip"
}
module "local_tracker_sink_plugin" {
  source                                         = "../../../modules/aws/hm_local_tracker_sink_plugin"
  snowflake_kafka_connector_version              = "2.2.1"
  bc_fips_version                                = "1.0.2.4"
  bcpkix_fips_version                            = "1.0.7"
  confluent_kafka_connect_avro_converter_version = "7.6.1"
  local_dir_path                                 = "data/amazon-msk/plugins"
  local_file_name                                = local.tracker_sink_plugin_file_name
}
module "production_tracker_sink_plugin" {
  providers       = { aws = aws.production }
  source          = "../../../modules/aws/hm_amazon_s3_object"
  s3_bucket_name  = module.production_amazon_hm_production_s3_bucket.name
  s3_key          = "amazon-msk/plugins/${local.tracker_sink_plugin_file_name}"
  local_file_path = module.local_tracker_sink_plugin.local_file_path
}
module "production_hm_amazon_msk_tracker_sink_plugin" {
  providers                = { aws = aws.production }
  source                   = "../../../modules/aws/hm_amazon_msk_plugin"
  amazon_msk_plugin_name   = "tracker-sink-plugin"
  s3_bucket_arn            = module.production_amazon_hm_production_s3_bucket.arn
  amazon_msk_plugin_s3_key = module.production_tracker_sink_plugin.s3_key
}
locals {
  production_tracker_sink_connector_name = "ProductionTrackerSinkConnector"
}

module "production_hm_amazon_msk_tracker_sink_connector_iam" {
  providers                 = { aws = aws.production }
  source                    = "../../../modules/aws/hm_amazon_msk_connector_iam"
  amazon_msk_connector_name = local.production_tracker_sink_connector_name
  amazon_msk_arn            = module.production_hm_amazon_msk_cluster.arn
  msk_plugin_s3_bucket_name = module.production_amazon_hm_production_s3_bucket.name
  msk_log_s3_bucket_name    = module.production_amazon_hm_production_s3_bucket.name
  environment               = var.environment
  team                      = var.team
}
data "aws_secretsmanager_secret" "tracker_snowflake_secret" {
  name = "hm/snowflake/production_hm_kafka_db/product/read_write"
}
data "aws_secretsmanager_secret_version" "tracker_snowflake_secret_version" {
  secret_id = data.aws_secretsmanager_secret.tracker_snowflake_secret.id
}
module "production_hm_amazon_msk_tracker_sink_connector" {
  providers                            = { aws = aws.production }
  source                               = "../../../modules/aws/hm_amazon_msk_connector"
  amazon_msk_connector_name            = local.production_tracker_sink_connector_name
  kafka_connect_version                = "2.7.1"
  amazon_msk_plugin_arn                = module.production_hm_amazon_msk_tracker_sink_plugin.arn
  amazon_msk_plugin_revision           = module.production_hm_amazon_msk_tracker_sink_plugin.latest_revision
  amazon_msk_connector_iam_role_arn    = module.production_hm_amazon_msk_tracker_sink_connector_iam.arn
  amazon_msk_cluster_bootstrap_servers = module.production_hm_amazon_msk_cluster.bootstrap_servers
  snowflake_user_name                  = jsondecode(data.aws_secretsmanager_secret_version.tracker_snowflake_secret_version.secret_string)["user_name"]
  snowflake_private_key                = jsondecode(data.aws_secretsmanager_secret_version.tracker_snowflake_secret_version.secret_string)["private_key"]
  snowflake_private_key_passphrase     = jsondecode(data.aws_secretsmanager_secret_version.tracker_snowflake_secret_version.secret_string)["private_key_passphrase"]
  snowflake_role_name                  = "HM_PRODUCTION_HM_KAFKA_DB_PRODUCT_READ_WRITE_ROLE"
  msk_log_s3_bucket_name               = module.production_amazon_hm_production_s3_bucket.name
  msk_log_s3_key                       = "amazon-msk/connectors/${local.production_tracker_sink_connector_name}"
  kafka_topic_name                     = "production.tracker.analytic-events.avro"
  snowflake_database_name              = "PRODUCTION_HM_KAFKA_DB"
  snowflake_schema_name                = "PRODUCT"
  environment                          = var.environment
  team                                 = var.team
}
