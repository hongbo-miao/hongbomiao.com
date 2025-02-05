variable "mimir_service_account_name" {
  type = string
}
variable "mimir_namespace" {
  type = string
}
variable "mimir_alertmanager_s3_bucket_name" {
  type = string
}
variable "mimir_block_s3_bucket_name" {
  type = string
}
variable "mimir_ruler_s3_bucket_name" {
  type = string
}
variable "amazon_eks_cluster_oidc_provider" {
  type = string
}
variable "amazon_eks_cluster_oidc_provider_arn" {
  type = string
}
variable "environment" {
  type = string
}
variable "team" {
  type = string
}
