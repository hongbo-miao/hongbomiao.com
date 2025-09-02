module "git_repository_hongbomiao_com" {
  source          = "../../../modules/argo-cd/argo_cd_private_git_repository"
  repository      = "git@github.com:hongbo-miao/hongbomiao.com.git"
  ssh_private_key = var.git_repository_hongbomiao_com_ssh_private_key
}
