terraform {
  required_providers {
    aws = {
      source = "hashicorp/aws"
    }
  }
}

# https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/instance
resource "aws_instance" "hm_ec2_instance" {
  ami           = var.ec2_instance_ami
  instance_type = var.ec2_instance_type
  tags = {
    Name = var.ec2_instance_name
  }
}
