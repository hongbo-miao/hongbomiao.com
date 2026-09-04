variable "zitadel_access_token" {
  type      = string
  sensitive = true
}
variable "google_oidc_client_secret" {
  type      = string
  sensitive = true
  default   = ""
}
