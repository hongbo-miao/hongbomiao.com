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
      version = "5.24.0"
    }
    # https://registry.terraform.io/providers/hashicorp/awscc/latest
    awscc = {
      source  = "hashicorp/awscc"
      version = "0.63.0"
    }
  }
  required_version = ">= 1.6"
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
module "hm_trino" {
  source                         = "./modules/hm_amazon_emr_cluster"
  amazon_emr_cluster_name        = "hm-trino"
  amazon_emr_version             = "emr-6.14.0"
  applications                   = ["Trino"]
  primary_instance_type          = "r7a.xlarge"
  core_instance_type             = "r7a.2xlarge"
  core_target_on_demand_capacity = 1
  bootstrap_set_up_script_s3_uri = "s3://hongbomiao-bucket/amazon-emr/hm-amazon-emr-cluster-trino/bootstrap-actions/set_up.sh"
  configurations = [
    {
      Classification : "delta-defaults",
      Properties : {
        "delta.enabled" : "true"
      }
    },
    {
      Classification : "trino-connector-delta",
      Properties : {
        "hive.metastore" : "glue"
      }
    },
    {
      Classification : "trino-connector-postgresql",
      Properties : {
        connection-url : "jdbc:postgresql://${jsondecode(data.aws_secretsmanager_secret_version.hm_rds_secret_version.secret_string)["postgres_host"]}:${jsondecode(data.aws_secretsmanager_secret_version.hm_rds_secret_version.secret_string)["postgres_port"]}/${jsondecode(data.aws_secretsmanager_secret_version.hm_rds_secret_version.secret_string)["postgres_db"]}",
        connection-user : jsondecode(data.aws_secretsmanager_secret_version.hm_rds_secret_version.secret_string)["postgres_user"],
        connection-password : jsondecode(data.aws_secretsmanager_secret_version.hm_rds_secret_version.secret_string)["postgres_password"]
      }
    }
  ]
  aws_iam_role = "arn:aws:iam::272394222652:role/service-role/AmazonEMR-ServiceRole-hm"
  environment  = var.environment
  team         = var.team
}
module "hm_trino_task_instance_fleet" {
  source                    = "./modules/hm_amazon_emr_cluster_task_instance_fleet"
  amazon_emr_cluster_id     = module.hm_trino.id
  task_instance_type        = "r7a.2xlarge"
  task_target_spot_capacity = 7
}
data "aws_instance" "hm_trino_primary_node_ec2_instance" {
  filter {
    name   = "private-dns-name"
    values = [module.hm_trino.master_public_dns]
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
module "hm_sedona_emr" {
  source                         = "./modules/hm_amazon_emr_cluster"
  amazon_emr_cluster_name        = "hm-sedona"
  amazon_emr_version             = "emr-6.14.0"
  applications                   = ["Hadoop", "Hive", "JupyterEnterpriseGateway", "Spark"]
  primary_instance_type          = "r7a.2xlarge"
  core_instance_type             = "r7a.2xlarge"
  core_target_on_demand_capacity = 1
  bootstrap_set_up_script_s3_uri = "s3://hongbomiao-bucket/amazon-emr/hm-amazon-emr-cluster-sedona/bootstrap-actions/set_up.sh"
  configurations = [
    {
      Classification : "delta-defaults",
      Properties : {
        "delta.enabled" : "true"
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
        "spark.yarn.dist.jars" : "/usr/lib/spark/jars/sedona-spark-shaded-3.4_2.12-1.5.0.jar,/usr/lib/spark/jars/geotools-wrapper-1.5.0-28.2.jar",
        "spark.serializer" : "org.apache.spark.serializer.KryoSerializer",
        "spark.kryo.registrator" : "org.apache.sedona.core.serde.SedonaKryoRegistrator",
        "spark.sql.extensions" : "org.apache.sedona.viz.sql.SedonaVizExtensions,org.apache.sedona.sql.SedonaSqlExtensions"
      }
    }
  ]
  aws_iam_role = "arn:aws:iam::272394222652:role/service-role/AmazonEMR-ServiceRole-hm"
  environment  = var.environment
  team         = var.team
}
module "hm_sedona_emr_task_instance_fleet" {
  source                    = "./modules/hm_amazon_emr_cluster_task_instance_fleet"
  amazon_emr_cluster_id     = module.hm_sedona_emr.id
  task_instance_type        = "r7a.2xlarge"
  task_target_spot_capacity = 1
}
module "hm_sedona_emr_managed_scaling_policy" {
  source                = "./modules/hm_amazon_emr_managed_scaling_policy"
  amazon_emr_cluster_id = module.hm_sedona_emr.id
  max_capacity_units    = 10
}
module "hm_sedona_emr_studio" {
  source                      = "./modules/hm_amazon_emr_studio"
  amazon_emr_studio_name      = "hm-sedona-emr-studio"
  s3_bucket                   = "hongbomiao-bucket"
  s3_uri                      = "s3://hongbomiao-bucket/amazon-emr/studio/hm-sedona-emr-studio"
  engine_security_group_id    = "sg-xxxxxxxxxxxxxxxxx"
  workspace_security_group_id = "sg-xxxxxxxxxxxxxxxxx"
  vpc_id                      = "vpc-xxxxxxxxxxxxxxxxx"
  subnet_ids                  = ["subnet-xxxxxxxxxxxxxxxxx"]
  environment                 = var.environment
  team                        = var.team
}

# AWS Glue DataBrew job
# AWS Glue DataBrew job - ADS-B 2x Flight Trace
module "hm_glue_databrew_job_write_csv_to_parquet_adsb_2x_flight_trace_data" {
  source                     = "./modules/hm_aws_glue_databrew_job"
  aws_glue_databrew_job_name = "hm-write-csv-to-parquet-adsb-2x-flight-trace-data"
  source_name                = "adsb-2x-flight-trace"
  recipe_version             = "1.0"
  input_s3_bucket            = "hongbomiao-bucket"
  output_s3_bucket           = "hongbomiao-bucket"
  output_s3_dir              = "data/raw-parquet/adsb_2x_flight_trace_data/"
  environment                = var.environment
  team                       = var.team
}

# AWS Glue job
# AWS Glue job - ADS-B 2x Flight Trace
module "hm_glue_job_write_parquet_to_delta_table_adsb_2x_flight_trace_data_script" {
  source           = "./modules/hm_amazon_s3_object"
  amazon_s3_bucket = "hongbomiao-bucket"
  amazon_s3_key    = "aws-glue/spark-scripts/hm_write_parquet_to_delta_table_adsb_2x_flight_trace_data.py"
  local_file_path  = "./data/aws-glue/spark-scripts/src/hm_write_parquet_to_delta_table_adsb_2x_flight_trace_data.py"
}
module "hm_glue_job_write_parquet_to_delta_table_adsb_2x_flight_trace_data" {
  source                  = "./modules/hm_aws_glue_job"
  aws_glue_job_name       = "hm_write_parquet_to_delta_table_adsb_2x_flight_trace_data"
  spark_script_s3_uri     = "s3://hongbomiao-bucket/aws-glue/spark-scripts/hm_write_parquet_to_delta_table_adsb_2x_flight_trace_data.py"
  spark_worker_type       = "G.1X"
  spark_worker_max_number = 50
  timeout_min             = 360
  aws_iam_role            = "arn:aws:iam::272394222652:role/service-role/AWSGlueServiceRole-hm"
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
  aws_iam_role            = "arn:aws:iam::272394222652:role/service-role/AWSGlueServiceRole-hm"
  environment             = var.environment
  team                    = var.team
}
module "hm_glue_crawler_motor_data" {
  source                        = "./modules/hm_aws_glue_crawler"
  aws_glue_crawler_name         = "hm-delta-lake-crawler-iot"
  aws_glue_crawler_delta_tables = ["s3://hongbomiao-bucket/delta-tables/motor_data/"]
  aws_glue_database             = "hm_delta_db"
  aws_iam_role                  = "arn:aws:iam::272394222652:role/service-role/AWSGlueServiceRole-hm"
  environment                   = var.environment
  team                          = var.team
}
