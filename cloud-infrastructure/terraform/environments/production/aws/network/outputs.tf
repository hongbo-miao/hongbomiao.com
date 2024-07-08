output "hm_amazon_vpc_id" {
  value = data.aws_vpc.hm_amazon_vpc.id
}
output "hm_amazon_vpc_subnets_ids" {
  value = data.aws_subnets.hm_amazon_vpc_subnets.ids
}
