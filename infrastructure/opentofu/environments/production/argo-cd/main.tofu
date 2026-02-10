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
  destination_namespaces = [
    "${var.environment}-*",
    "kube-system",
    "skypilot-system",
    local.argo_cd_namespace
  ]
  namespace_resource_whitelist = [
    { group = "*", kind = "*" },
  ]
  cluster_resource_whitelist = [
    { group = "", kind = "Namespace" },
    { group = "", kind = "PersistentVolume" },
    { group = "admissionregistration.k8s.io", kind = "MutatingWebhookConfiguration" },
    { group = "admissionregistration.k8s.io", kind = "ValidatingWebhookConfiguration" },
    { group = "apiextensions.k8s.io", kind = "CustomResourceDefinition" },
    { group = "apiregistration.k8s.io", kind = "APIService" },
    { group = "cert-manager.io", kind = "ClusterIssuer" },
    { group = "networking.k8s.io", kind = "IngressClass" },
    { group = "rbac.authorization.k8s.io", kind = "ClusterRole" },
    { group = "rbac.authorization.k8s.io", kind = "ClusterRoleBinding" },
    { group = "storage.k8s.io", kind = "CSIDriver" },
    { group = "storage.k8s.io", kind = "StorageClass" },
  ]
}
module "hm_project_bootstrap" {
  source            = "../../../modules/argo-cd/argo_cd_application"
  application_name  = "${local.hm_project}-bootstrap"
  argo_cd_project   = local.hm_project
  argo_cd_namespace = local.argo_cd_namespace
  repository        = local.hm_git_repository
  path              = local.hm_project
  include           = "{**/application-set.yaml}"
  depends_on = [
    module.hm_git_repository,
    module.hm_project
  ]
}
