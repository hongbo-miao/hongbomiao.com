# Amazon VPC
data "aws_vpc" "hm_amazon_vpc" {
  default = true
}
data "aws_subnets" "hm_amazon_vpc_private_subnets" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.hm_amazon_vpc.id]
  }
  tags = {
    Tier = "Private"
  }
}
data "aws_subnets" "hm_amazon_vpc_public_subnets" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.hm_amazon_vpc.id]
  }
  tags = {
    Tier = "Public"
  }
}
