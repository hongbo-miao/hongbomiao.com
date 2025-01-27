variable "loki_service_account_name" {
  type = string
}
variable "loki_namespace" {
  type = string
}
variable "loki_admin_data_s3_bucket_name" {
  type = string
}
variable "loki_chunk_data_s3_bucket_name" {
  type = string
}
variable "loki_ruler_data_s3_bucket_name" {
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
