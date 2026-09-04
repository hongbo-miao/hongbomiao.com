terraform {
  required_providers {
    zitadel = {
      source = "registry.opentofu.org/zitadel/zitadel"
    }
  }
}

# https://search.opentofu.org/provider/zitadel/zitadel/latest/docs/resources/default_login_policy
resource "zitadel_default_login_policy" "main" {
  user_login                    = true
  allow_register                = false
  allow_external_idp            = true
  idps                          = var.idps
  passwordless_type             = "PASSWORDLESS_TYPE_ALLOWED"
  default_redirect_uri          = ""
  force_mfa                     = false
  force_mfa_local_only          = false
  hide_password_reset           = false
  ignore_unknown_usernames      = false
  allow_domain_discovery        = true
  password_check_lifetime       = "864000s"
  external_login_check_lifetime = "864000s"
  mfa_init_skip_lifetime        = "2592000s"
  second_factor_check_lifetime  = "64800s"
  multi_factor_check_lifetime   = "43200s"
  second_factors                = ["SECOND_FACTOR_TYPE_OTP", "SECOND_FACTOR_TYPE_U2F"]
  multi_factors                 = ["MULTI_FACTOR_TYPE_U2F_WITH_VERIFICATION"]
}
