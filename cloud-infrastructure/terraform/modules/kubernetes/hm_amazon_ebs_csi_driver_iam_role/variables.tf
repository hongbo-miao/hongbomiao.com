variable "common_tags" {
  type = map(string)
}
variable "amazon_eks_cluster_name" {
  type = string
}
variable "amazon_eks_cluster_oidc_provider" {
  type = string
}
variable "amazon_eks_cluster_oidc_provider_arn" {
  type = string
}
variable "amazon_eks_cluster_name_hash" {
  type = string
}
