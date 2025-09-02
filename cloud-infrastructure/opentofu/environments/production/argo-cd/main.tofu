locals {
  argo_cd_namespace = "${var.environment}-hm-argo-cd"
}

# Project: hm
locals {
  hm_project        = "${var.environment}-hm"
  hm_git_repository = "git@github.com:hongbo-miao/hongbomiao.com.git"
}
module "hm_git_repository" {
  source          = "../../../modules/argo-cd/argo_cd_private_git_repository"
  repository      = local.hm_git_repository
  ssh_private_key = var.git_repository_hongbomiao_com_ssh_private_key
}
module "hm_project" {
  source            = "../../../modules/argo-cd/argo_cd_project"
  project_name      = local.hm_project
  argo_cd_namespace = local.argo_cd_namespace
}
module "hm_project_bootstrap" {
  source            = "../../../modules/argo-cd/argo_cd_application"
  application_name  = "${local.hm_project}-bootstrap"
  argo_cd_project   = local.hm_project
  argo_cd_namespace = local.argo_cd_namespace
  repository        = local.hm_git_repository
  path              = local.hm_project
  depends_on = [
    module.hm_git_repository,
    module.hm_project
  ]
}
