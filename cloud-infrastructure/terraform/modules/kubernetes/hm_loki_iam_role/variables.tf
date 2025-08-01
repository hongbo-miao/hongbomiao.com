variable "common_tags" {
  type = map(string)
}
variable "loki_service_account_name" {
  type = string
}
variable "loki_namespace" {
  type = string
}
variable "loki_admin_s3_bucket_name" {
  type = string
}
variable "loki_chunk_s3_bucket_name" {
  type = string
}
variable "loki_ruler_s3_bucket_name" {
  type = string
}
variable "amazon_eks_cluster_oidc_provider" {
  type = string
}
variable "amazon_eks_cluster_oidc_provider_arn" {
  type = string
}
