data "zitadel_org" "default" {
  id = "xxxxxxxxxxxxxxxxxx"
}

# Google OAuth
module "google_idp" {
  source        = "../../../modules/zitadel/zitadel_idp_google"
  name          = "Google"
  client_id     = "xxxxxxxxxxxx-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx.apps.googleusercontent.com"
  client_secret = var.google_oidc_client_secret
}
module "default_login_policy" {
  source = "../../../modules/zitadel/zitadel_default_login_policy"
  idps   = [module.google_idp.id]
}

# My App
module "my_app_project" {
  source = "../../../modules/zitadel/zitadel_project"
  name   = "my-app"
  org_id = data.zitadel_org.default.id
}
module "my_app_admin_role" {
  source       = "../../../modules/zitadel/zitadel_project_role"
  org_id       = data.zitadel_org.default.id
  project_id   = module.my_app_project.id
  role_key     = "admin"
  display_name = "admin"
}
module "my_app_user_role" {
  source       = "../../../modules/zitadel/zitadel_project_role"
  org_id       = data.zitadel_org.default.id
  project_id   = module.my_app_project.id
  role_key     = "user"
  display_name = "user"
}
module "my_app_application" {
  source                    = "../../../modules/zitadel/zitadel_application_oidc"
  org_id                    = data.zitadel_org.default.id
  project_id                = module.my_app_project.id
  name                      = "my-app"
  redirect_uris             = ["https://my-app.hongbomiao.com/auth/callback"]
  post_logout_redirect_uris = ["https://my-app.hongbomiao.com/"]
}
