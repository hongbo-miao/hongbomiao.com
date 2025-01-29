variable "tempo_service_account_name" {
  type = string
}
variable "tempo_namespace" {
  type = string
}
variable "tempo_admin_s3_bucket_name" {
  type = string
}
variable "tempo_trace_s3_bucket_name" {
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
