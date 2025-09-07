variable "common_tags" {
  type = map(string)
}
variable "redpanda_console_service_account_name" {
  type = string
}
variable "redpanda_console_namespace" {
  type = string
}
variable "amazon_eks_cluster_oidc_provider" {
  type = string
}
variable "amazon_eks_cluster_oidc_provider_arn" {
  type = string
}
