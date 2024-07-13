variable "amazon_route53_hosted_zone_id" {
  type = string
}
variable "amazon_route53_hosted_zone_name" {
  type = string
}
variable "amazon_vpc_id" {
  type = string
}
variable "amazon_vpc_private_subnet_ids" {
  type = list(string)
}
variable "environment" {
  type = string
}
variable "team" {
  type = string
}
