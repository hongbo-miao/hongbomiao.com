variable "common_tags" {
  type = map(string)
}
variable "external_dns_service_account_name" {
  type = string
}
variable "external_dns_namespace" {
  type = string
}
variable "amazon_eks_cluster_oidc_provider" {
  type = string
}
variable "amazon_eks_cluster_oidc_provider_arn" {
  type = string
}
variable "amazon_route53_hosted_zone_id" {
  type = string
}
variable "amazon_eks_cluster_name_hash" {
  type = string
}
