variable "common_tags" {
  type = map(string)
}
variable "label_studio_service_account_name" {
  type = string
}
variable "label_studio_namespace" {
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
