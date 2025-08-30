terraform {
  required_providers {
    harbor = {
      source = "goharbor/harbor"
    }
  }
}

# https://registry.terraform.io/providers/goharbor/harbor/latest/docs/resources/config_auth
resource "harbor_config_auth" "google_auth" {
  auth_mode          = "oidc_auth"
  primary_auth_mode  = var.primary_auth_mode
  oidc_name          = "google"
  oidc_endpoint      = "https://accounts.google.com"
  oidc_client_id     = var.oidc_client_id
  oidc_client_secret = var.oidc_client_secret
  oidc_scope         = "openid,email"
  oidc_user_claim    = "name"
  oidc_verify_cert   = true
  oidc_auto_onboard  = true
}
