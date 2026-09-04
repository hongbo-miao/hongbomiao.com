terraform {
  required_providers {
    zitadel = {
      source = "registry.opentofu.org/zitadel/zitadel"
    }
  }
}

# https://search.opentofu.org/provider/zitadel/zitadel/latest/docs/resources/idp_google
resource "zitadel_idp_google" "main" {
  name                = var.name
  client_id           = var.client_id
  client_secret       = var.client_secret
  scopes              = ["openid", "profile", "email"]
  is_creation_allowed = true
  is_linking_allowed  = true
  auto_linking        = "AUTO_LINKING_OPTION_EMAIL"
  is_auto_creation    = true
  is_auto_update      = true
}
