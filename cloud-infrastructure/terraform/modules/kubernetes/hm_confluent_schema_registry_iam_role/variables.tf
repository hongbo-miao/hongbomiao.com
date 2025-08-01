variable "common_tags" {
  type = map(string)
}
variable "confluent_schema_registry_service_account_nickname" {
  type = string
}
variable "confluent_schema_registry_service_account_name" {
  type = string
}
variable "confluent_schema_registry_namespace" {
  type = string
}
variable "amazon_eks_cluster_oidc_provider" {
  type = string
}
variable "amazon_eks_cluster_oidc_provider_arn" {
  type = string
}
variable "amazon_msk_arn" {
  type = string
}
