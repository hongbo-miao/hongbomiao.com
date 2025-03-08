variable "prefect_worker_service_account_name" {
  type = string
}
variable "prefect_worker_namespace" {
  type = string
}
variable "iot_data_s3_bucket_name" {
  type = string
}
variable "aws_glue_database_names" {
  type = list(string)
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
