terraform {
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

module "hm_ec2_module" {
  source            = "./modules/hm_ec2"
  ec2_instance_name = "hm-ec2-instance"
}
