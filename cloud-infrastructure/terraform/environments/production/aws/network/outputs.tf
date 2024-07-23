output "hm_amazon_vpc_id" {
  value = data.aws_vpc.hm_amazon_vpc.id
}
output "hm_amazon_vpc_ipv4_cidr_block" {
  value = data.aws_vpc.hm_amazon_vpc.cidr_block
}
output "hm_amazon_vpc_private_subnets_ids" {
  value = data.aws_subnets.hm_amazon_vpc_private_subnets.ids
}
output "hm_amazon_vpc_public_subnets_ids" {
  value = data.aws_subnets.hm_amazon_vpc_public_subnets.ids
}
