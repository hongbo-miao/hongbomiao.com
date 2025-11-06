variable "common_tags" {
  type = map(string)
}
variable "mlflow_service_account_name" {
  type = string
}
variable "mlflow_namespace" {
  type = string
}
variable "amazon_eks_cluster_oidc_provider" {
  type = string
}
variable "amazon_eks_cluster_oidc_provider_arn" {
  type = string
}
variable "s3_bucket_name" {
  type = string
}
