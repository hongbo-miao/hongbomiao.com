module "harbor_config_system" {
  source                       = "../../../modules/harbor/hm_harbor_config_system"
  project_creation_restriction = "adminonly"
}

# Config
data "aws_secretsmanager_secret" "hm_harbor_google_client_secret" {
  name = "${var.environment}-hm-harbor-google-client"
}
data "aws_secretsmanager_secret_version" "hm_harbor_google_client_secret_version" {
  secret_id = data.aws_secretsmanager_secret.hm_harbor_google_client_secret.id
}
module "harbor_config_google_auth" {
  source             = "../../../modules/harbor/hm_harbor_config_google_auth"
  primary_auth_mode  = true
  oidc_client_id     = "xxxxxxxxxxxx-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx.apps.googleusercontent.com"
  oidc_client_secret = jsondecode(data.aws_secretsmanager_secret_version.hm_harbor_google_client_secret_version.secret_string)["secret"]
}

# Registries
module "harbor_registry_docker_hub" {
  source        = "../../../modules/harbor/hm_harbor_registry"
  provider_name = "docker-hub"
  name          = "docker-hub"
  endpoint_url  = "https://hub.docker.com"
}
module "harbor_registry_microsoft_artifact_registry" {
  source        = "../../../modules/harbor/hm_harbor_registry"
  provider_name = "docker-registry"
  name          = "microsoft-artifact-registry"
  endpoint_url  = "https://mcr.microsoft.com"
}

# Garbage Collection
module "harbor_garbage_collection" {
  source        = "../../../modules/harbor/hm_harbor_garbage_collection"
  schedule      = "0 7 9 1 * *" # 9:07 AM on the 1st day of every month (UTC)
  worker_number = 2
}

# Projects
# Projects - docker-hub-proxy-cache
module "harbor_project_docker_hub_proxy_cache" {
  source      = "../../../modules/harbor/hm_harbor_project"
  name        = "docker-hub-proxy-cache"
  public      = true
  registry_id = module.harbor_registry_docker_hub.id
}
# Projects - microsoft-artifact-registry-proxy-cache
module "harbor_project_microsoft_artifact_registry_proxy_cache" {
  source      = "../../../modules/harbor/hm_harbor_project"
  name        = "microsoft-artifact-registry-proxy-cache"
  public      = true
  registry_id = module.harbor_registry_microsoft_artifact_registry.id
}
# Projects - hongbomiao
module "harbor_project_hm" {
  source = "../../../modules/harbor/hm_harbor_project"
  name   = "hongbomiao"
  public = false
}

# Robots
data "aws_secretsmanager_secret" "hm_harbor_hm_kubernetes_robot_secret" {
  provider = aws.production
  name     = "${var.environment}-hm-harbor-hm-kubernetes-robot"
}
data "aws_secretsmanager_secret_version" "hm_harbor_hm_kubernetes_robot_secret_version" {
  provider  = aws.production
  secret_id = data.aws_secretsmanager_secret.hm_harbor_hm_kubernetes_robot_secret.id
}
module "harbor_robot_account_hm_kubernetes_robot" {
  source        = "../../../modules/harbor/hm_harbor_robot_account"
  name          = "hm-kubernetes-robot"
  actions       = ["pull"]
  project_names = [module.harbor_project_hm.name]
  secret        = jsondecode(data.aws_secretsmanager_secret_version.hm_harbor_hm_kubernetes_robot_secret_version.secret_string)["secret"]
}
