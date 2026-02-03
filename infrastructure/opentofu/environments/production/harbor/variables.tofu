variable "harbor_username" {
  type = string
}
variable "harbor_password" {
  type      = string
  sensitive = true
}
variable "harbor_google_client_secret" {
  type      = string
  sensitive = true
}
variable "hm_kubernetes_robot_secret" {
  type      = string
  sensitive = true
}
variable "hm_application_robot_secret" {
  type      = string
  sensitive = true
}
