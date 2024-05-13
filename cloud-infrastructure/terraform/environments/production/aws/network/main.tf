# Amazon VPC
data "aws_vpc" "hm_amazon_vpc" {
  default = true
}
data "aws_subnets" "hm_amazon_vpc_subnets" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.hm_amazon_vpc.id]
  }
}
