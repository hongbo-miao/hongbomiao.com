variable "common_tags" {
  type = map(string)
}
variable "starrocks_service_account_name" {
  type = string
}
variable "starrocks_namespace" {
  type = string
}
variable "iot_data_s3_bucket_name" {
  type = string
}
variable "amazon_eks_cluster_oidc_provider" {
  type = string
}
variable "amazon_eks_cluster_oidc_provider_arn" {
  type = string
}
