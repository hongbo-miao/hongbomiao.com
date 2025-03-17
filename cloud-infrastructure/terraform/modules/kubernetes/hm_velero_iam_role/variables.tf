variable "velero_service_account_name" {
  type = string
}
variable "velero_namespace" {
  type = string
}
variable "s3_bucket_name" {
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
