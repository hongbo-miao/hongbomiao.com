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
      version = "5.15.0"
    }
  }
  required_version = ">= 1.5"
}

provider "aws" {
  region = "us-west-2"
}

module "hm_amazon_ec2_module" {
  source            = "./modules/hm_amazon_ec2"
  ec2_instance_name = "hm-amazon-ec2-instance"
  ec2_instance_ami  = "ami-0c79a55dda52434da" # Ubuntu Server 22.04 LTS (HVM), SSD Volume Type (64-bit (Arm))
  ec2_instance_type = "t2.nano"
}
module "hm_aws_glue_crawler_module" {
  source                        = "./modules/hm_aws_glue_crawler"
  aws_glue_crawler_name         = "hm-delta-lake-crawler-iot"
  aws_glue_crawler_delta_tables = ["s3://hongbomiao-bucket/delta-tables/motor/"]
  aws_glue_database             = "hm_delta_db"
  aws_glue_crawler_role         = "arn:aws:iam::272394222652:role/service-role/AWSGlueServiceRole-hm"
  environment                   = var.environment
  team                          = var.team
}
