terraform {
  required_providers {
    zitadel = {
      source = "registry.opentofu.org/zitadel/zitadel"
    }
  }
}

# https://search.opentofu.org/provider/zitadel/zitadel/latest/docs/resources/application_oidc
resource "zitadel_application_oidc" "main" {
  org_id                      = var.org_id
  project_id                  = var.project_id
  name                        = var.name
  redirect_uris               = var.redirect_uris
  post_logout_redirect_uris   = var.post_logout_redirect_uris
  response_types              = ["OIDC_RESPONSE_TYPE_CODE"]
  grant_types                 = ["OIDC_GRANT_TYPE_AUTHORIZATION_CODE", "OIDC_GRANT_TYPE_REFRESH_TOKEN"]
  app_type                    = "OIDC_APP_TYPE_USER_AGENT"
  auth_method_type            = "OIDC_AUTH_METHOD_TYPE_NONE"
  version                     = "OIDC_VERSION_1_0"
  dev_mode                    = var.dev_mode
  id_token_role_assertion     = true
  access_token_role_assertion = true
}
