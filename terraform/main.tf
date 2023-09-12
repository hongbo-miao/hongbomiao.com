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
      version = "5.16.2"
    }
  }
  required_version = ">= 1.5"
}

provider "aws" {
  region = "us-west-2"
}

# Amazon EC2
module "hm_amazon_ec2_module" {
  source            = "./modules/hm_amazon_ec2"
  ec2_instance_name = "hm-amazon-ec2-instance"
  ec2_instance_ami  = "ami-0c79a55dda52434da" # Ubuntu Server 22.04 LTS (HVM), SSD Volume Type (64-bit (Arm))
  ec2_instance_type = "t2.nano"
}

# Amazon EMR
module "hm_amazon_emr_cluster" {
  source                  = "./modules/hm_amazon_emr_cluster"
  amazon_emr_cluster_name = "hm-emr-cluster-trino"
  amazon_emr_version      = "emr-6.13.0"
  primary_instance_type   = "r5.xlarge"
  core_instance_type      = "r5.2xlarge"
  core_instance_count     = 1
  aws_iam_role            = "arn:aws:iam::272394222652:role/service-role/AmazonEMR-ServiceRole-hm"
  environment             = var.environment
  team                    = var.team
}
module "hm_amazon_emr_cluster_task_instance_group" {
  source                = "./modules/hm_amazon_emr_cluster_task_instance_group"
  amazon_emr_cluster_id = module.hm_amazon_emr_cluster.id
  task_instance_type    = "r5.2xlarge"
  task_instance_count   = 1
}

# AWS Glue
module "hm_aws_glue_job" {
  source                   = "./modules/hm_aws_glue_job"
  aws_glue_job_name        = "hm_write_parquet_to_delta_lake_motor"
  spark_script_s3_location = "s3://hongbomiao-bucket/aws-glue/spark-scripts/hm_write_parquet_to_delta_lake_motor.py"
  aws_iam_role             = "arn:aws:iam::272394222652:role/service-role/AWSGlueServiceRole-hm"
  environment              = var.environment
  team                     = var.team
}
module "hm_aws_glue_crawler_module" {
  source                        = "./modules/hm_aws_glue_crawler"
  aws_glue_crawler_name         = "hm-delta-lake-crawler-iot"
  aws_glue_crawler_delta_tables = ["s3://hongbomiao-bucket/delta-tables/motor/"]
  aws_glue_database             = "hm_delta_db"
  aws_iam_role                  = "arn:aws:iam::272394222652:role/service-role/AWSGlueServiceRole-hm"
  environment                   = var.environment
  team                          = var.team
}
