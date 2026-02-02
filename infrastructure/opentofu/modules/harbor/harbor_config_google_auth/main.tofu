terraform {
  required_providers {
    harbor = {
      source = "registry.opentofu.org/goharbor/harbor"
    }
  }
}

# https://search.opentofu.org/provider/goharbor/harbor/latest/docs/resources/config_auth
resource "harbor_config_auth" "google_auth" {
  auth_mode          = "oidc_auth"
  primary_auth_mode  = var.primary_auth_mode
  oidc_name          = "google"
  oidc_endpoint      = "https://accounts.google.com"
  oidc_client_id     = var.oidc_client_id
  oidc_client_secret = var.oidc_client_secret
  oidc_scope         = "email,openid"
  oidc_user_claim    = "email"
  oidc_verify_cert   = true
  oidc_auto_onboard  = true
}
