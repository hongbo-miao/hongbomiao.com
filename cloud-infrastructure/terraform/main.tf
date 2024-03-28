terraform {
  # backend "local" {
  #   path = "terraform.tfstate"
  # }
  backend "s3" {
    region = "us-west-2"
    bucket = "hongbomiao-bucket"
    key    = "terraform/terraform.tfstate"
  }
  required_providers {
    # https://registry.terraform.io/providers/hashicorp/aws/latest
    aws = {
      source  = "hashicorp/aws"
      version = "5.43.0"
    }
    # https://registry.terraform.io/providers/hashicorp/awscc/latest
    awscc = {
      source  = "hashicorp/awscc"
      version = "0.72.1"
    }
  }
  # terraform version
  required_version = ">= 1.7"
}

provider "aws" {
  region = "us-west-2"
}

# Amazon EC2
module "hm_ec2_module" {
  source            = "./modules/hm_amazon_ec2"
  ec2_instance_name = "hm-ec2-instance"
  ec2_instance_ami  = "ami-0c79a55dda52434da" # Ubuntu Server 22.04 LTS (HVM), SSD Volume Type (64-bit (Arm))
  ec2_instance_type = "t2.nano"
}

# Amazon EMR
# Amazon EMR - Trino
data "aws_secretsmanager_secret" "hm_rds_secret" {
  name = "hm-iot-rds/hm_iot_db/readonly"
}
data "aws_secretsmanager_secret_version" "hm_rds_secret_version" {
  secret_id = data.aws_secretsmanager_secret.hm_rds_secret.id
}
module "hm_trino_s3_set_up_script" {
  source           = "./modules/hm_amazon_s3_object"
  amazon_s3_bucket = "hongbomiao-bucket"
  amazon_s3_key    = "amazon-emr/hm-amazon-emr-cluster-trino/bootstrap-actions/set_up.sh"
  local_file_path  = "./data/amazon-emr/hm-amazon-emr-cluster-trino/bootstrap-actions/set_up.sh"
}
module "hm_trino_emr" {
  source                                     = "./modules/hm_amazon_emr_cluster"
  amazon_emr_cluster_name                    = "hm-trino"
  amazon_emr_version                         = "emr-7.0.0"
  applications                               = ["Trino"]
  primary_instance_target_on_demand_capacity = 1
  primary_instance_weighted_capacity         = 1
  primary_instance_type                      = "r7g.xlarge"
  core_instance_target_on_demand_capacity    = 1
  core_instance_weighted_capacity            = 1
  core_instance_type                         = "r7g.xlarge"
  bootstrap_set_up_script_s3_uri             = module.hm_trino_s3_set_up_script.uri
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
          "connection-url": "jdbc:postgresql://${jsondecode(data.aws_secretsmanager_secret_version.hm_rds_secret_version.secret_string)["host"]}:${jsondecode(data.aws_secretsmanager_secret_version.hm_rds_secret_version.secret_string)["port"]}/${jsondecode(data.aws_secretsmanager_secret_version.hm_rds_secret_version.secret_string)["db"]}",
          "connection-user": "${jsondecode(data.aws_secretsmanager_secret_version.hm_rds_secret_version.secret_string)["user"]}",
          "connection-password": "${jsondecode(data.aws_secretsmanager_secret_version.hm_rds_secret_version.secret_string)["password"]}"
        }
      }
    ]
  EOF
  iam_role_arn                               = "arn:aws:iam::272394222652:role/service-role/AmazonEMR-ServiceRole-hm"
  environment                                = var.environment
  team                                       = var.team
}
module "hm_trino_task_instance_fleet" {
  source                             = "./modules/hm_amazon_emr_cluster_task_instance_fleet"
  amazon_emr_cluster_id              = module.hm_trino_emr.id
  task_instance_target_spot_capacity = 1
  task_instance_weighted_capacity    = 1
  task_instance_type                 = "r7g.xlarge"
}
data "aws_instance" "hm_trino_primary_node_ec2_instance" {
  filter {
    name   = "private-dns-name"
    values = [module.hm_trino_emr.master_public_dns]
  }
}
module "hm_route_53_record" {
  source                        = "./modules/hm_amazon_route_53"
  amazon_route_53_record_name   = "hm-emr-trino"
  amazon_route_53_record_values = [data.aws_instance.hm_trino_primary_node_ec2_instance.private_ip]
}

# Amazon EMR - Apache Sedona
module "hm_sedona_s3_set_up_script" {
  source           = "./modules/hm_amazon_s3_object"
  amazon_s3_bucket = "hongbomiao-bucket"
  amazon_s3_key    = "amazon-emr/clusters/hm-amazon-emr-cluster-sedona/bootstrap-actions/set_up.sh"
  local_file_path  = "./data/amazon-emr/hm-amazon-emr-cluster-sedona/bootstrap-actions/set_up.sh"
}
module "hm_sedona_s3_validate_python_version_script" {
  source           = "./modules/hm_amazon_s3_object"
  amazon_s3_bucket = "hongbomiao-bucket"
  amazon_s3_key    = "amazon-emr/clusters/hm-amazon-emr-cluster-sedona/steps/validate_python_version.py"
  local_file_path  = "./data/amazon-emr/hm-amazon-emr-cluster-sedona/steps/validate_python_version.py"
}
module "hm_sedona_s3_set_up_jupyterlab_script" {
  source           = "./modules/hm_amazon_s3_object"
  amazon_s3_bucket = "hongbomiao-bucket"
  amazon_s3_key    = "amazon-emr/clusters/hm-amazon-emr-cluster-sedona/steps/set_up_jupyterlab.sh"
  local_file_path  = "./data/amazon-emr/hm-amazon-emr-cluster-sedona/steps/set_up_jupyterlab.sh"
}
module "hm_sedona_emr" {
  source                                     = "./modules/hm_amazon_emr_cluster"
  amazon_emr_cluster_name                    = "hm-sedona"
  amazon_emr_version                         = "emr-7.0.0"
  applications                               = ["Hadoop", "Hive", "JupyterEnterpriseGateway", "Livy", "Spark"]
  primary_instance_target_on_demand_capacity = 1
  primary_instance_weighted_capacity         = 1
  primary_instance_type                      = "r7g.xlarge"
  core_instance_target_on_demand_capacity    = 1
  core_instance_weighted_capacity            = 1
  core_instance_type                         = "r7g.xlarge"
  bootstrap_set_up_script_s3_uri             = module.hm_sedona_s3_set_up_script.uri
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
module "hm_sedona_emr_task_instance_fleet" {
  source                             = "./modules/hm_amazon_emr_cluster_task_instance_fleet"
  amazon_emr_cluster_id              = module.hm_sedona_emr.id
  task_instance_target_spot_capacity = 2
  task_instance_weighted_capacity    = 2
  task_instance_type                 = "r7g.2xlarge"
}
module "hm_sedona_emr_managed_scaling_policy" {
  source                = "./modules/hm_amazon_emr_managed_scaling_policy"
  amazon_emr_cluster_id = module.hm_sedona_emr.id
  max_capacity_units    = 60
}
data "aws_instance" "hm_sedona_emr_primary_node" {
  filter {
    name   = "private-dns-name"
    values = [module.hm_sedona_emr.master_public_dns]
  }
}
module "hm_sedona_route_53" {
  source                        = "./modules/hm_amazon_route_53"
  amazon_route_53_record_name   = "hm-sedona"
  amazon_route_53_record_values = [data.aws_instance.hm_sedona_emr_primary_node.private_ip]
}

module "hm_sedona_emr_studio_iam" {
  source                 = "./modules/hm_amazon_emr_studio_iam"
  amazon_emr_studio_name = "hm-sedona-emr-studio"
  s3_bucket              = "hongbomiao-bucket"
  environment            = var.environment
  team                   = var.team
}
module "hm_sedona_emr_studio" {
  source                 = "./modules/hm_amazon_emr_studio"
  amazon_emr_studio_name = "hm-sedona-emr-studio"
  s3_uri                 = "s3://hongbomiao-bucket/amazon-emr/studio/hm-sedona-emr-studio"
  iam_role_arn           = module.hm_sedona_emr_studio_iam.arn
  environment            = var.environment
  team                   = var.team
}

# AWS Glue DataBrew job
# AWS Glue DataBrew recipe job - ADS-B 2x Flight Trace - Write CSV to Parquet
module "hm_glue_databrew_recipe_job_write_adsb_2x_flight_trace_csv_to_parquet_iam" {
  source                         = "./modules/hm_aws_glue_databrew_iam"
  aws_glue_databrew_job_nickname = "write-adsb-csv-to-parquet"
  input_s3_bucket                = "hongbomiao-bucket"
  output_s3_bucket               = "hongbomiao-bucket"
  environment                    = var.environment
  team                           = var.team
}
module "hm_glue_databrew_recipe_job_write_adsb_2x_flight_trace_csv_to_parquet_dataset" {
  source                         = "./modules/hm_aws_glue_databrew_dataset_adsb_raw_data"
  count                          = length(var.adsb_2x_flight_trace_raw_data_dates)
  aws_glue_databrew_dataset_name = "adsb-2x-flight-trace-dataset-raw-data-${replace(var.adsb_2x_flight_trace_raw_data_dates[count.index], "/", "-")}"
  input_s3_bucket                = "hongbomiao-bucket"
  input_s3_dir                   = "data/raw/adsb_2x_flight_trace_data/${var.adsb_2x_flight_trace_raw_data_dates[count.index]}/"
  environment                    = var.environment
  team                           = var.team
}
module "hm_glue_databrew_recipe_job_write_adsb_2x_flight_trace_csv_to_parquet" {
  source                            = "./modules/hm_aws_glue_databrew_recipe_job"
  count                             = length(var.adsb_2x_flight_trace_raw_data_dates)
  aws_glue_databrew_recipe_job_name = "hm-write-adsb-2x-flight-trace-csv-to-parquet-${replace(var.adsb_2x_flight_trace_raw_data_dates[count.index], "/", "-")}"
  aws_glue_databrew_dataset_name    = "adsb-2x-flight-trace-dataset-raw-data-${replace(var.adsb_2x_flight_trace_raw_data_dates[count.index], "/", "-")}"
  recipe_name                       = "adsb-2x-flight-trace-recipe"
  recipe_version                    = "1.0"
  spark_worker_max_number           = 24
  timeout_min                       = 1440
  output_s3_bucket                  = "hongbomiao-bucket"
  output_s3_dir                     = "data/raw-parquet/adsb_2x_flight_trace_data/${var.adsb_2x_flight_trace_raw_data_dates[count.index]}/"
  output_max_file_number            = 24
  iam_role_arn                      = module.hm_glue_databrew_recipe_job_write_adsb_2x_flight_trace_csv_to_parquet_iam.arn
  environment                       = var.environment
  team                              = var.team
  depends_on = [
    module.hm_glue_databrew_recipe_job_write_adsb_2x_flight_trace_csv_to_parquet_dataset
  ]
}

# AWS Glue DataBrew profile job - ADS-B 2x Flight Trace - Profile Parquet
module "hm_glue_databrew_profile_job_profile_adsb_2x_flight_trace_raw_parquet_iam" {
  source                         = "./modules/hm_aws_glue_databrew_iam"
  aws_glue_databrew_job_nickname = "profile-adsb-raw-parquet"
  input_s3_bucket                = "hongbomiao-bucket"
  output_s3_bucket               = "hongbomiao-bucket"
  environment                    = var.environment
  team                           = var.team
}
module "hm_glue_databrew_profile_job_profile_adsb_2x_flight_trace_raw_parquet_dataset" {
  source                         = "./modules/hm_aws_glue_databrew_dataset_adsb_raw_parquet"
  count                          = length(var.adsb_2x_flight_trace_raw_parquet_dates)
  aws_glue_databrew_dataset_name = "adsb-2x-flight-trace-dataset-raw-parquet-${replace(var.adsb_2x_flight_trace_raw_parquet_dates[count.index], "/", "-")}"
  input_s3_bucket                = "hongbomiao-bucket"
  input_s3_dir                   = "data/raw-parquet/adsb_2x_flight_trace_data/${var.adsb_2x_flight_trace_raw_parquet_dates[count.index]}/"
  environment                    = var.environment
  team                           = var.team
}
module "hm_glue_databrew_profile_job_profile_adsb_2x_flight_trace_raw_parquet" {
  source                             = "./modules/hm_aws_glue_databrew_profile_job"
  count                              = length(var.adsb_2x_flight_trace_raw_parquet_dates)
  aws_glue_databrew_profile_job_name = "hm-profile-adsb-2x-flight-trace-raw-parquet-${replace(var.adsb_2x_flight_trace_raw_parquet_dates[count.index], "/", "-")}"
  aws_glue_databrew_dataset_name     = "adsb-2x-flight-trace-dataset-raw-parquet-${replace(var.adsb_2x_flight_trace_raw_parquet_dates[count.index], "/", "-")}"
  spark_worker_max_number            = 24
  timeout_min                        = 1440
  output_s3_bucket                   = "hongbomiao-bucket"
  output_s3_dir                      = "aws-glue-databrew/profile-results/"
  iam_role_arn                       = module.hm_glue_databrew_profile_job_profile_adsb_2x_flight_trace_raw_parquet_iam.arn
  environment                        = var.environment
  team                               = var.team
  depends_on = [
    module.hm_glue_databrew_profile_job_profile_adsb_2x_flight_trace_raw_parquet_dataset
  ]
}

# AWS Glue job
# AWS Glue job - ADS-B 2x Flight Trace
module "hm_glue_job_write_parquet_to_delta_table_adsb_2x_flight_trace_data_script_iam" {
  source                = "./modules/hm_aws_glue_iam"
  aws_glue_job_nickname = "write-adsb-parquet-to-delta-table"
  input_s3_bucket       = "hongbomiao-bucket"
  output_s3_bucket      = "hongbomiao-bucket"
  environment           = var.environment
  team                  = var.team
}
module "hm_glue_job_write_parquet_to_delta_table_adsb_2x_flight_trace_data_script" {
  source           = "./modules/hm_amazon_s3_object"
  amazon_s3_bucket = "hongbomiao-bucket"
  amazon_s3_key    = "aws-glue/spark-scripts/hm_write_parquet_to_delta_table_adsb_2x_flight_trace_data.py"
  local_file_path  = "./data/aws-glue/spark-scripts/src/hm_write_parquet_to_delta_table_adsb_2x_flight_trace_data.py"
}
module "hm_glue_job_write_parquet_to_delta_table_adsb_2x_flight_trace_data" {
  source                  = "./modules/hm_aws_glue_job"
  aws_glue_job_name       = "hm_write_parquet_to_delta_table_adsb_2x_flight_trace_data"
  spark_script_s3_uri     = module.hm_glue_job_write_parquet_to_delta_table_adsb_2x_flight_trace_data_script.uri
  spark_worker_type       = "G.1X"
  spark_worker_max_number = 900
  timeout_min             = 360
  iam_role_arn            = module.hm_glue_job_write_parquet_to_delta_table_adsb_2x_flight_trace_data_script_iam.arn
  environment             = var.environment
  team                    = var.team
}

# AWS Glue job - Motor
module "hm_glue_job_write_parquet_to_delta_table_motor_data_script" {
  source           = "./modules/hm_amazon_s3_object"
  amazon_s3_bucket = "hongbomiao-bucket"
  amazon_s3_key    = "aws-glue/spark-scripts/hm_write_parquet_to_delta_table_motor_data.py"
  local_file_path  = "./data/aws-glue/spark-scripts/src/hm_write_parquet_to_delta_table_motor_data.py"
}
module "hm_glue_job_write_parquet_to_delta_table_motor_data" {
  source                  = "./modules/hm_aws_glue_job"
  aws_glue_job_name       = "hm_write_parquet_to_delta_lake_motor_data"
  spark_script_s3_uri     = "s3://hongbomiao-bucket/aws-glue/spark-scripts/hm_write_parquet_to_delta_lake_motor_data.py"
  spark_worker_type       = "G.1X"
  spark_worker_max_number = 20
  timeout_min             = 360
  iam_role_arn            = "arn:aws:iam::272394222652:role/service-role/AWSGlueServiceRole-hm"
  environment             = var.environment
  team                    = var.team
}
module "hm_glue_crawler_motor_data" {
  source                        = "./modules/hm_aws_glue_crawler"
  aws_glue_crawler_name         = "hm-delta-lake-crawler-iot"
  aws_glue_crawler_delta_tables = ["s3://hongbomiao-bucket/delta-tables/motor_data/"]
  aws_glue_database             = "hm_delta_db"
  iam_role_arn                  = "arn:aws:iam::272394222652:role/service-role/AWSGlueServiceRole-hm"
  environment                   = var.environment
  team                          = var.team
}
