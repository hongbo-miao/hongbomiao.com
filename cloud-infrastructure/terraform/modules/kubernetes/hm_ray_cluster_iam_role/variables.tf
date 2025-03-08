variable "ray_cluster_service_account_name" {
  type = string
}
variable "ray_cluster_namespace" {
  type = string
}
variable "amazon_eks_cluster_oidc_provider" {
  type = string
}
variable "amazon_eks_cluster_oidc_provider_arn" {
  type = string
}
variable "mlflow_s3_bucket_name" {
  type = string
}
variable "iot_data_s3_bucket_name" {
  type = string
}
variable "environment" {
  type = string
}
variable "team" {
  type = string
}
