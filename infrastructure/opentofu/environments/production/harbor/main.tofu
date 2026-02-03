module "harbor_config_system" {
  source                       = "../../../modules/harbor/harbor_config_system"
  project_creation_restriction = "adminonly"
}

# Config
module "harbor_config_google_auth" {
  source             = "../../../modules/harbor/harbor_config_google_auth"
  primary_auth_mode  = true
  oidc_client_id     = "xxxxxxxxxxxx-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx.apps.googleusercontent.com"
  oidc_client_secret = var.harbor_google_client_secret
}

# Registries
module "harbor_registry_docker_hub" {
  source        = "../../../modules/harbor/harbor_registry"
  provider_name = "docker-hub"
  name          = "docker-hub"
  endpoint_url  = "https://hub.docker.com"
}
module "harbor_registry_microsoft_artifact_registry" {
  source        = "../../../modules/harbor/harbor_registry"
  provider_name = "docker-registry"
  name          = "microsoft-artifact-registry"
  endpoint_url  = "https://mcr.microsoft.com"
}

# Garbage Collection
module "harbor_garbage_collection" {
  source        = "../../../modules/harbor/harbor_garbage_collection"
  schedule      = "0 7 9 1 * *" # 9:07 AM on the 1st day of every month (UTC)
  worker_number = 2
}

# Project
# Project - docker-hub-proxy-cache
module "harbor_project_docker_hub_proxy_cache" {
  source      = "../../../modules/harbor/harbor_project"
  name        = "docker-hub-proxy-cache"
  public      = true
  registry_id = module.harbor_registry_docker_hub.id
}
# Project - microsoft-artifact-registry-proxy-cache
module "harbor_project_microsoft_artifact_registry_proxy_cache" {
  source      = "../../../modules/harbor/harbor_project"
  name        = "microsoft-artifact-registry-proxy-cache"
  public      = true
  registry_id = module.harbor_registry_microsoft_artifact_registry.id
}
# Project - hongbomiao
module "harbor_project_hm_application" {
  source = "../../../modules/harbor/harbor_project"
  name   = "hm-application"
  public = false
}

# Robot
# Robot - hm-kubernetes-robot
module "harbor_robot_account_hm_kubernetes_robot" {
  source        = "../../../modules/harbor/harbor_robot_account"
  name          = "hm-kubernetes-robot"
  actions       = ["pull"]
  project_names = [module.harbor_project_hm_application.name]
  secret        = var.hm_kubernetes_robot_secret
}
# Robot - hm-application-project-robot
module "harbor_robot_account_hm_application_robot" {
  source        = "../../../modules/harbor/harbor_robot_account"
  name          = "hm-application-project-robot"
  actions       = ["pull", "push"]
  project_names = [module.harbor_project_hm_application.name]
  secret        = var.hm_application_robot_secret
}
