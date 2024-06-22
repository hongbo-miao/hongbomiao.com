variable "kubernetes_namespace" {
  type = string
}
variable "labels" {
  type    = map(string)
  default = {}
}
