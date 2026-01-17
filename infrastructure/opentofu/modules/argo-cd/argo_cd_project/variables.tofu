variable "project_name" {
  type = string
}
variable "argo_cd_namespace" {
  type = string
}
variable "namespace_resource_whitelist" {
  type = list(object({
    group = string
    kind  = string
  }))
}
variable "cluster_resource_whitelist" {
  type = list(object({
    group = string
    kind  = string
  }))
}
