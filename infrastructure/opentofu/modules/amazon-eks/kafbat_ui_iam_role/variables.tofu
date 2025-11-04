variable "common_tags" {
  type = map(string)
}
variable "kafbat_ui_service_account_name" {
  type = string
}
variable "kafbat_ui_namespace" {
  type = string
}
variable "amazon_eks_cluster_oidc_provider" {
  type = string
}
variable "amazon_eks_cluster_oidc_provider_arn" {
  type = string
}
