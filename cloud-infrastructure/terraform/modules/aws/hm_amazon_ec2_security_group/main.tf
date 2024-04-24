terraform {
  required_providers {
    aws = {
      source = "hashicorp/aws"
    }
  }
}

# https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/security_group
resource "aws_security_group" "hm_amazon_ec2_security_group" {
  name   = var.amazon_ec2_security_group_name
  vpc_id = var.amazon_vpc_id
  egress {
    from_port = 0
    to_port   = 0
    protocol  = "-1"
    cidr_blocks = [
      "0.0.0.0/0",
    ]
  }
  tags = {
    Environment = var.environment
    Team        = var.team
    Name        = var.amazon_ec2_security_group_name
  }
}
