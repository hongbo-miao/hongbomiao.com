variable "common_tags" {
  type = map(string)
}
variable "litellm_service_account_name" {
  type = string
}
variable "litellm_namespace" {
  type = string
}
variable "amazon_eks_cluster_oidc_provider" {
  type = string
}
variable "amazon_eks_cluster_oidc_provider_arn" {
  type = string
}
