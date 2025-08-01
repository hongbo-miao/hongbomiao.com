data "terraform_remote_state" "production_aws_network_terraform_remote_state" {
  backend = "s3"
  config = {
    region = "us-west-2"
    bucket = "hm-terraform-bucket"
    key    = "production/aws/network/terraform.tfstate"
  }
}
data "terraform_remote_state" "production_aws_data_terraform_remote_state" {
  backend = "s3"
  config = {
    region = "us-west-2"
    bucket = "hm-terraform-bucket"
    key    = "production/aws/data/terraform.tfstate"
  }
}

# Tracker Kafka Manager
locals {
  kafka_manager_name = "hm-${var.environment}-kafka-manager"
}
module "tracker_kafka_manager_key" {
  providers   = { aws = aws.production }
  source      = "../../../../modules/aws/hm_amazon_ec2_key_pair"
  key_name    = "${local.kafka_manager_name}-key"
  public_key  = var.tracker_kafka_manager_public_key
  common_tags = var.common_tags
}
module "tracker_kafka_manager_instance_profile" {
  providers          = { aws = aws.production }
  source             = "../../../../modules/aws/hm_kafka_manager_ec2_instance_profile"
  kafka_manager_name = local.kafka_manager_name
  amazon_msk_arn     = data.terraform_remote_state.production_aws_data_terraform_remote_state.outputs.tracker_kafka_cluster_arn
  common_tags        = var.common_tags
}
module "tracker_kafka_manager_instance" {
  providers                    = { aws = aws.production }
  source                       = "../../../../modules/aws/hm_amazon_ec2_instance"
  instance_name                = local.kafka_manager_name
  instance_ami                 = "ami-06ffa14aad0078bd0" # ubuntu/images/hvm-ssd-gp3/ubuntu-noble-24.04-arm64-server-20240701.1
  instance_type                = "t4g.medium"
  instance_profile             = module.tracker_kafka_manager_instance_profile.name
  key_name                     = module.tracker_kafka_manager_key.name
  amazon_vpc_subnet_id         = data.terraform_remote_state.production_aws_data_terraform_remote_state.outputs.hm_amazon_vpc_private_subnets_ids[0]
  amazon_ec2_security_group_id = "sg-xxxxxxxxxxxxxxxxx"
  common_tags                  = var.common_tags
}


# Amazon EMR
# Amazon EMR - Trino
data "aws_secretsmanager_secret" "hm_rds_secret" {
  provider = aws.production
  name     = "hm/amazon-rds/production_hm_iot_db/public/read_only"
}
data "aws_secretsmanager_secret_version" "hm_rds_secret_version" {
  provider  = aws.production
  secret_id = data.aws_secretsmanager_secret.hm_rds_secret.id
}
locals {
  trino_cluster_name = "hm-trino"
}
module "s3_object_hm_trino_set_up_script" {
  providers       = { aws = aws.production }
  source          = "../../../../modules/aws/hm_amazon_s3_object"
  s3_bucket_name  = data.terraform_remote_state.production_aws_data_terraform_remote_state.outputs.hm_amazon_vpc_private_subnets_ids
  s3_key          = "amazon-emr/clusters/${local.trino_cluster_name}/bootstrap-actions/set_up.sh"
  local_file_path = "files/amazon-emr/clusters/${local.trino_cluster_name}/bootstrap-actions/set_up.sh"
}
module "hm_trino_emr" {
  providers                                  = { aws = aws.production }
  source                                     = "../../../../modules/aws/hm_amazon_emr_cluster"
  amazon_emr_cluster_name                    = local.trino_cluster_name
  amazon_emr_version                         = "emr-7.5.0"
  applications                               = ["Trino"]
  primary_instance_target_on_demand_capacity = 3
  primary_instance_type                      = "c7g.4xlarge"
  primary_instance_ebs_volume_size_gb        = 16
  core_instance_target_on_demand_capacity    = 4
  core_instance_weighted_capacity            = 4
  core_instance_type                         = "m7g.4xlarge"
  core_instance_ebs_volume_size_gb           = 16
  bootstrap_set_up_script_s3_uri             = module.s3_object_hm_trino_set_up_script.uri
  configurations_json_string                 = <<EOF
    [
      {
        "Classification": "hive-site",
        "Properties": {
          "hive.metastore.client.factory.class": "com.amazonaws.glue.catalog.metastore.AWSGlueDataCatalogHiveClientFactory"
        }
      },
      {
        "Classification": "trino-connector-hive",
        "Properties": {
          "hive.metastore": "glue"
        }
      },
      {
        "Classification": "trino-config",
        "Properties": {
          "retry-policy": "TASK",
          "exchange.compression-codec": "LZ4",
          "query.low-memory-killer.delay": "0s",
          "query.remote-task.max-error-duration": "1m",
          "task.low-memory-killer.policy": "total-reservation-on-blocked-nodes"
        }
      },
      {
        "Classification": "trino-exchange-manager",
        "Properties": {
          "exchange-manager.name": "filesystem",
          "exchange.base-directories": "s3://${data.terraform_remote_state.production_aws_data_terraform_remote_state.outputs.hm_production_bucket_name}/amazon-emr/clusters/${local.trino_cluster_name}/exchange-spooling",
          "exchange.s3.region": "us-west-2"
        }
      },
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
  placement_group_config = [
    {
      instance_role      = "MASTER"
      placement_strategy = "SPREAD"
    }
  ]
  iam_role_arn = "arn:aws:iam::272394222652:role/service-role/AmazonEMR-ServiceRole-hm"
  common_tags  = var.common_tags
  depends_on = [
    module.s3_object_hm_trino_set_up_script
  ]
}
module "hm_trino_task_instance_fleet" {
  providers                          = { aws = aws.production }
  source                             = "../../../../modules/aws/hm_amazon_emr_cluster_task_instance_fleet"
  amazon_emr_cluster_id              = module.hm_trino_emr.id
  task_instance_target_spot_capacity = 28
  task_instance_configs = [
    {
      instance_type      = "m7g.4xlarge"
      weighted_capacity  = 4
      ebs_volume_size_gb = 16
    },
    {
      instance_type      = "m7a.4xlarge"
      weighted_capacity  = 4
      ebs_volume_size_gb = 16
    },
    {
      instance_type      = "m7i.4xlarge"
      weighted_capacity  = 4
      ebs_volume_size_gb = 16
    },
    {
      instance_type      = "m6g.4xlarge"
      weighted_capacity  = 4
      ebs_volume_size_gb = 16
    },
    {
      instance_type      = "m6a.4xlarge"
      weighted_capacity  = 4
      ebs_volume_size_gb = 16
    },
    {
      instance_type      = "m6i.4xlarge"
      weighted_capacity  = 4
      ebs_volume_size_gb = 16
    }
  ]
}
data "aws_instance" "hm_trino_primary_node_ec2_instance" {
  provider = aws.production
  filter {
    name   = "private-dns-name"
    values = [module.hm_trino_emr.master_public_dns]
  }
}
module "hm_route_53_record" {
  providers                     = { aws = aws.production }
  source                        = "../../../../modules/aws/hm_amazon_route_53"
  amazon_route_53_record_name   = "hm-emr-trino"
  amazon_route_53_record_values = [data.aws_instance.hm_trino_primary_node_ec2_instance.private_ip]
}

# Amazon EMR - Apache Sedona
module "s3_object_hm_sedona_set_up_script" {
  providers       = { aws = aws.production }
  source          = "../../../../modules/aws/hm_amazon_s3_object"
  s3_bucket_name  = data.terraform_remote_state.production_aws_data_terraform_remote_state.outputs.hm_production_bucket_name
  s3_key          = "amazon-emr/clusters/hm-amazon-emr-cluster-sedona/bootstrap-actions/set_up.sh"
  local_file_path = "files/amazon-emr/hm-amazon-emr-cluster-sedona/bootstrap-actions/set_up.sh"
}
module "hm_sedona_cluster" {
  providers                                  = { aws = aws.production }
  source                                     = "../../../../modules/aws/hm_amazon_emr_cluster"
  amazon_emr_cluster_name                    = "hm-sedona"
  amazon_emr_version                         = "emr-7.2.0"
  applications                               = ["Hadoop", "Hive", "JupyterEnterpriseGateway", "Livy", "Spark"]
  primary_instance_target_on_demand_capacity = 1
  primary_instance_type                      = "r7g.xlarge"
  primary_instance_ebs_volume_size_gb        = 16
  core_instance_target_on_demand_capacity    = 1
  core_instance_weighted_capacity            = 1
  core_instance_type                         = "r7g.2xlarge"
  core_instance_ebs_volume_size_gb           = 16
  bootstrap_set_up_script_s3_uri             = module.s3_object_hm_sedona_set_up_script.uri
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
  common_tags                                = var.common_tags
  depends_on = [
    module.s3_object_hm_sedona_set_up_script
  ]
}
module "hm_sedona_cluster_task_instance_fleet" {
  providers                          = { aws = aws.production }
  source                             = "../../../../modules/aws/hm_amazon_emr_cluster_task_instance_fleet"
  amazon_emr_cluster_id              = module.hm_sedona_cluster.id
  task_instance_target_spot_capacity = 2
  task_instance_configs = [
    {
      instance_type      = "r7g.2xlarge"
      weighted_capacity  = 2
      ebs_volume_size_gb = 16
    },
    {
      instance_type      = "r6g.2xlarge"
      weighted_capacity  = 2
      ebs_volume_size_gb = 16
    }
  ]
}
module "hm_sedona_cluster_managed_scaling_policy" {
  providers             = { aws = aws.production }
  source                = "../../../../modules/aws/hm_amazon_emr_managed_scaling_policy"
  amazon_emr_cluster_id = module.hm_sedona_cluster.id
  max_capacity_units    = 60
}
data "aws_instance" "hm_sedona_cluster_primary_node" {
  provider = aws.production
  filter {
    name   = "private-dns-name"
    values = [module.hm_sedona_cluster.master_public_dns]
  }
}
module "hm_sedona_route_53" {
  providers                     = { aws = aws.production }
  source                        = "../../../../modules/aws/hm_amazon_route_53"
  amazon_route_53_record_name   = "hm-sedona"
  amazon_route_53_record_values = [data.aws_instance.hm_sedona_cluster_primary_node.private_ip]
}

module "hm_sedona_cluster_studio_iam" {
  providers              = { aws = aws.production }
  source                 = "../../../../modules/aws/hm_amazon_emr_studio_iam_role"
  amazon_emr_studio_name = "hm-sedona-emr-studio"
  s3_bucket_name         = data.terraform_remote_state.production_aws_data_terraform_remote_state.outputs.hm_production_bucket_name
  common_tags            = var.common_tags
}
module "hm_sedona_cluster_studio" {
  providers              = { aws = aws.production }
  source                 = "../../../../modules/aws/hm_amazon_emr_studio"
  amazon_emr_studio_name = "hm-sedona-emr-studio"
  s3_uri                 = "s3://hm-production-bucket/amazon-emr/studio/hm-sedona-emr-studio"
  iam_role_arn           = module.hm_sedona_cluster_studio_iam.arn
  common_tags            = var.common_tags
}

# AWS Glue DataBrew job
# AWS Glue DataBrew recipe job - ADS-B 2x Flight Trace - Write CSV to Parquet
module "hm_glue_databrew_recipe_job_write_adsb_2x_flight_trace_csv_to_parquet_iam" {
  providers                      = { aws = aws.production }
  source                         = "../../../../modules/aws/hm_aws_glue_databrew_iam_role"
  aws_glue_databrew_job_nickname = "write-adsb-csv-to-parquet"
  input_s3_bucket_name           = data.terraform_remote_state.production_aws_data_terraform_remote_state.outputs.hm_production_bucket_name
  output_s3_bucket_name          = data.terraform_remote_state.production_aws_data_terraform_remote_state.outputs.hm_production_bucket_name
  common_tags                    = var.common_tags
}
module "hm_glue_databrew_recipe_job_write_adsb_2x_flight_trace_csv_to_parquet_dataset" {
  providers                      = { awscc = awscc.production }
  source                         = "../../../../modules/aws/hm_aws_glue_databrew_dataset_adsb_raw_data"
  for_each                       = toset(var.adsb_2x_flight_trace_raw_data_dates)
  aws_glue_databrew_dataset_name = "adsb-2x-flight-trace-dataset-raw-data-${replace(each.value, "/", "-")}"
  input_s3_bucket_name           = data.terraform_remote_state.production_aws_data_terraform_remote_state.outputs.hm_production_bucket_name
  input_s3_key                   = "data/raw/adsb_2x_flight_trace_data/${each.value}/"
  common_tags                    = var.common_tags
}
module "hm_glue_databrew_recipe_job_write_adsb_2x_flight_trace_csv_to_parquet" {
  providers                         = { awscc = awscc.production }
  source                            = "../../../../modules/aws/hm_aws_glue_databrew_recipe_job"
  for_each                          = toset(var.adsb_2x_flight_trace_raw_data_dates)
  aws_glue_databrew_recipe_job_name = "hm-write-adsb-2x-flight-trace-csv-to-parquet-${replace(each.value, "/", "-")}"
  aws_glue_databrew_dataset_name    = "adsb-2x-flight-trace-dataset-raw-data-${replace(each.value, "/", "-")}"
  recipe_name                       = "adsb-2x-flight-trace-recipe"
  recipe_version                    = "1.0"
  spark_worker_max_number           = 24
  timeout_min                       = 1440
  output_s3_bucket_name             = data.terraform_remote_state.production_aws_data_terraform_remote_state.outputs.hm_production_bucket_name
  output_s3_key                     = "data/raw-parquet/adsb_2x_flight_trace_data/${each.value}/"
  output_max_file_number            = 24
  iam_role_arn                      = module.hm_glue_databrew_recipe_job_write_adsb_2x_flight_trace_csv_to_parquet_iam.arn
  common_tags                       = var.common_tags
  depends_on = [
    module.hm_glue_databrew_recipe_job_write_adsb_2x_flight_trace_csv_to_parquet_dataset
  ]
}

# AWS Glue DataBrew profile job - ADS-B 2x Flight Trace - Profile Parquet
module "hm_glue_databrew_profile_job_profile_adsb_2x_flight_trace_raw_parquet_iam" {
  providers                      = { aws = aws.production }
  source                         = "../../../../modules/aws/hm_aws_glue_databrew_iam_role"
  aws_glue_databrew_job_nickname = "profile-adsb-raw-parquet"
  input_s3_bucket_name           = data.terraform_remote_state.production_aws_data_terraform_remote_state.outputs.hm_production_bucket_name
  output_s3_bucket_name          = data.terraform_remote_state.production_aws_data_terraform_remote_state.outputs.hm_production_bucket_name
  common_tags                    = var.common_tags
}
module "hm_glue_databrew_profile_job_profile_adsb_2x_flight_trace_raw_parquet_dataset" {
  providers                      = { awscc = awscc.production }
  source                         = "../../../../modules/aws/hm_aws_glue_databrew_dataset_adsb_raw_parquet"
  for_each                       = toset(var.adsb_2x_flight_trace_raw_parquet_dates)
  aws_glue_databrew_dataset_name = "adsb-2x-flight-trace-dataset-raw-parquet-${replace(each.value, "/", "-")}"
  input_s3_bucket_name           = data.terraform_remote_state.production_aws_data_terraform_remote_state.outputs.hm_production_bucket_name
  input_s3_key                   = "data/raw-parquet/adsb_2x_flight_trace_data/${each.value}/"
  common_tags                    = var.common_tags
}
module "hm_glue_databrew_profile_job_profile_adsb_2x_flight_trace_raw_parquet" {
  providers                          = { awscc = awscc.production }
  source                             = "../../../../modules/aws/hm_aws_glue_databrew_profile_job"
  for_each                           = toset(var.adsb_2x_flight_trace_raw_parquet_dates)
  aws_glue_databrew_profile_job_name = "hm-profile-adsb-2x-flight-trace-raw-parquet-${replace(each.value, "/", "-")}"
  aws_glue_databrew_dataset_name     = "adsb-2x-flight-trace-dataset-raw-parquet-${replace(each.value, "/", "-")}"
  spark_worker_max_number            = 24
  timeout_min                        = 1440
  output_s3_bucket_name              = data.terraform_remote_state.production_aws_data_terraform_remote_state.outputs.hm_production_bucket_name
  output_s3_key                      = "aws-glue-databrew/profile-results/"
  iam_role_arn                       = module.hm_glue_databrew_profile_job_profile_adsb_2x_flight_trace_raw_parquet_iam.arn
  common_tags                        = var.common_tags
  depends_on = [
    module.hm_glue_databrew_profile_job_profile_adsb_2x_flight_trace_raw_parquet_dataset
  ]
}

# AWS Glue job
# AWS Glue job - ADS-B 2x Flight Trace
module "hm_glue_job_write_parquet_to_delta_table_adsb_2x_flight_trace_data_script_iam" {
  providers             = { aws = aws.production }
  source                = "../../../../modules/aws/hm_aws_glue_iam_role"
  aws_glue_job_nickname = "write-adsb-parquet-to-delta-table"
  input_s3_bucket_name  = data.terraform_remote_state.production_aws_data_terraform_remote_state.outputs.hm_production_bucket_name
  output_s3_bucket_name = data.terraform_remote_state.production_aws_data_terraform_remote_state.outputs.hm_production_bucket_name
  common_tags           = var.common_tags
}
module "hm_glue_job_write_parquet_to_delta_table_adsb_2x_flight_trace_data_script" {
  providers       = { aws = aws.production }
  source          = "../../../../modules/aws/hm_amazon_s3_object"
  s3_bucket_name  = data.terraform_remote_state.production_aws_data_terraform_remote_state.outputs.hm_production_bucket_name
  s3_key          = "aws-glue/spark-scripts/hm_write_parquet_to_delta_table_adsb_2x_flight_trace_data.py"
  local_file_path = "files/aws-glue/spark-scripts/src/hm_write_parquet_to_delta_table_adsb_2x_flight_trace_data.py"
}
module "hm_glue_job_write_parquet_to_delta_table_adsb_2x_flight_trace_data" {
  providers               = { aws = aws.production }
  source                  = "../../../../modules/aws/hm_aws_glue_job"
  aws_glue_job_name       = "hm_write_parquet_to_delta_table_adsb_2x_flight_trace_data"
  aws_glue_version        = "5.0"
  spark_script_s3_uri     = module.hm_glue_job_write_parquet_to_delta_table_adsb_2x_flight_trace_data_script.uri
  spark_worker_type       = "G.1X"
  spark_worker_max_number = 900
  spark_conf              = "--conf spark.sql.sources.partitionOverwriteMode=dynamic --conf spark.sql.parquet.compression.codec=zstd --conf spark.sql.extensions=io.delta.sql.DeltaSparkSessionExtension --conf spark.sql.catalog.spark_catalog=org.apache.spark.sql.delta.catalog.DeltaCatalog --conf spark.delta.logStore.class=org.apache.spark.sql.delta.storage.S3SingleDriverLogStore"
  timeout_min             = 360
  iam_role_arn            = module.hm_glue_job_write_parquet_to_delta_table_adsb_2x_flight_trace_data_script_iam.arn
  common_tags             = var.common_tags
}

# AWS Glue job - Motor
module "hm_glue_job_write_parquet_to_delta_table_motor_data_script" {
  providers       = { aws = aws.production }
  source          = "../../../../modules/aws/hm_amazon_s3_object"
  s3_bucket_name  = data.terraform_remote_state.production_aws_data_terraform_remote_state.outputs.hm_production_bucket_name
  s3_key          = "aws-glue/spark-scripts/hm_write_parquet_to_delta_table_motor_data.py"
  local_file_path = "files/aws-glue/spark-scripts/src/hm_write_parquet_to_delta_table_motor_data.py"
}
module "hm_glue_job_write_parquet_to_delta_table_motor_data" {
  providers               = { aws = aws.production }
  source                  = "../../../../modules/aws/hm_aws_glue_job"
  aws_glue_job_name       = "hm_write_parquet_to_delta_lake_motor_data"
  aws_glue_version        = "5.0"
  spark_script_s3_uri     = "s3://hm-production-bucket/aws-glue/spark-scripts/hm_write_parquet_to_delta_lake_motor_data.py"
  spark_worker_type       = "G.1X"
  spark_worker_max_number = 20
  spark_conf              = "--conf spark.sql.sources.partitionOverwriteMode=dynamic --conf spark.sql.parquet.compression.codec=zstd --conf spark.sql.extensions=io.delta.sql.DeltaSparkSessionExtension --conf spark.sql.catalog.spark_catalog=org.apache.spark.sql.delta.catalog.DeltaCatalog --conf spark.delta.logStore.class=org.apache.spark.sql.delta.storage.S3SingleDriverLogStore"
  timeout_min             = 360
  iam_role_arn            = "arn:aws:iam::272394222652:role/service-role/AWSGlueServiceRole-hm"
  common_tags             = var.common_tags
}
module "hm_glue_crawler_motor_data" {
  providers                     = { aws = aws.production }
  source                        = "../../../../modules/aws/hm_aws_glue_crawler"
  aws_glue_crawler_name         = "hm-delta-lake-crawler-iot"
  aws_glue_crawler_delta_tables = ["s3://hm-production-bucket/delta-tables/motor_data/"]
  aws_glue_database             = "production_motor_db"
  schedule                      = "cron(40 9 * * ? *)" # Every day at 9:40 UTC https://crontab.cronhub.io/
  iam_role_arn                  = "arn:aws:iam::272394222652:role/service-role/AWSGlueServiceRole-hm"
  common_tags                   = var.common_tags
}

# AWS Batch
module "hm_aws_batch_security_group" {
  providers                      = { aws = aws.production }
  source                         = "../../../../modules/aws/hm_aws_batch_security_group"
  amazon_ec2_security_group_name = "hm-aws-batch-security-group"
  amazon_vpc_id                  = data.terraform_remote_state.production_aws_network_terraform_remote_state.outputs.hm_amazon_vpc_id
  common_tags                    = var.common_tags
}
module "hm_aws_batch_compute_environment_iam" {
  providers                              = { aws = aws.production }
  source                                 = "../../../../modules/aws/hm_aws_batch_compute_environment_iam_role"
  aws_batch_compute_environment_nickname = "hm-batch-compute-env"
  common_tags                            = var.common_tags
}
module "hm_aws_batch_compute_environment" {
  providers                          = { aws = aws.production }
  source                             = "../../../../modules/aws/hm_aws_batch_compute_environment"
  aws_batch_compute_environment_name = "hm-aws-batch-compute-environment"
  amazon_ec2_security_group_ids      = [module.hm_aws_batch_security_group.id]
  amazon_vpc_subnet_ids              = data.terraform_remote_state.production_aws_data_terraform_remote_state.outputs.hm_production_bucket_name
  iam_role_arn                       = module.hm_aws_batch_compute_environment_iam.arn
  common_tags                        = var.common_tags
  depends_on = [
    module.hm_aws_batch_compute_environment_iam
  ]
}
module "hm_aws_batch_job_queue" {
  providers                          = { aws = aws.production }
  source                             = "../../../../modules/aws/hm_aws_batch_job_queue"
  aws_batch_job_queue_name           = "hm-aws-batch-queue"
  aws_batch_compute_environment_arns = [module.hm_aws_batch_compute_environment.arn]
  common_tags                        = var.common_tags
}
module "hm_aws_batch_job_definition_iam" {
  providers                         = { aws = aws.production }
  source                            = "../../../../modules/aws/hm_aws_batch_job_definition_iam_role"
  aws_batch_job_definition_nickname = "hm-batch-job-def"
  common_tags                       = var.common_tags
}
module "hm_aws_batch_job_definition" {
  providers                     = { aws = aws.production }
  source                        = "../../../../modules/aws/hm_aws_batch_job_definition"
  aws_batch_job_definition_name = "hm-aws-batch-definition"
  iam_role_arn                  = module.hm_aws_batch_job_definition_iam.arn
  common_tags                   = var.common_tags
  depends_on = [
    module.hm_aws_batch_job_definition_iam
  ]
}

# Amazon SageMaker
locals {
  amazon_sagemaker_notebook_instance_name = "hm-amazon-sagemaker-notebook"
}
module "hm_amazon_sagemaker_notebook_instance_iam" {
  providers                               = { aws = aws.production }
  source                                  = "../../../../modules/aws/hm_amazon_sagemaker_notebook_instance_iam_role"
  amazon_sagemaker_notebook_instance_name = local.amazon_sagemaker_notebook_instance_name
  common_tags                             = var.common_tags
}
module "hm_amazon_sagemaker_notebook_instance" {
  providers                               = { aws = aws.production }
  source                                  = "../../../../modules/aws/hm_amazon_sagemaker_notebook_instance"
  amazon_sagemaker_notebook_instance_name = local.amazon_sagemaker_notebook_instance_name
  iam_role_arn                            = module.hm_amazon_sagemaker_notebook_instance_iam.arn
  instance_type                           = "ml.g4dn.4xlarge"
  common_tags                             = var.common_tags
}
