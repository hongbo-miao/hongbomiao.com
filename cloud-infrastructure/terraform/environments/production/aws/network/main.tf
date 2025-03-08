# Amazon VPC
data "aws_vpc" "current" {
  provider = aws.production
  default  = true
}
data "aws_subnets" "hm_amazon_vpc_private_subnets" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.current.id]
  }
  tags = {
    Tier = "Private"
  }
}
data "aws_subnets" "hm_amazon_vpc_public_subnets" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.current.id]
  }
  tags = {
    Tier = "Public"
  }
}
