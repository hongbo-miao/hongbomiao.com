module "hm_kubernetes_cluster" {
  source                  = "../../../../modules/nebius/kubernetes_cluster"
  project_id              = var.project_id
  kubernetes_cluster_name = "hm-kubernetes-cluster"
  vpc_subnet_id           = var.vpc_subnet_id
  labels                  = var.common_labels
}
module "hm_kubernetes_node_group_dragon" {
  source         = "../../../../modules/nebius/kubernetes_node_group"
  parent_id      = module.hm_kubernetes_cluster.id
  name           = "node-group-dragon"
  min_node_count = 3
  max_node_count = 5
  platform       = "cpu-d3"
  preset         = "8vcpu-32gb"
  labels         = var.common_labels
}

# Argo CD
# Argo CD - Kubernetes namespace
module "kubernetes_namespace_hm_argo_cd" {
  source               = "../../../../modules/kubernetes/kubernetes_namespace"
  kubernetes_namespace = "${var.environment}-hm-argo-cd"
  labels = {
    "goldilocks.fairwinds.com/enabled" = "true"
  }
  depends_on = [
    module.hm_kubernetes_cluster
  ]
}
# Argo CD - Helm chart
module "argo_cd_helm_chart" {
  source              = "../../../../modules/kubernetes/helm_chart"
  repository          = "https://argoproj.github.io/argo-helm"
  chart_name          = "argo-cd"
  chart_version       = "8.3.1" # https://artifacthub.io/packages/helm/argo/argo-cd
  release_name        = "argo-cd"
  namespace           = module.kubernetes_namespace_hm_argo_cd.namespace
  my_values_yaml_path = "files/argo-cd/helm/my-values.yaml"
  depends_on = [
    module.kubernetes_namespace_hm_argo_cd
  ]
}
