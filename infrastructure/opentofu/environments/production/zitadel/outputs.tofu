output "my_app_oidc_client_id" {
  value     = module.my_app_application.client_id
  sensitive = true
}
output "my_app_project_id" {
  value = module.my_app_project.id
}
