# Organization
# Organization - hongbomiao
module "grafana_organization_hongbomiao" {
  source = "../../../modules/grafana/grafana_organization"
  name   = "hongbomiao"
}
locals {
  org_ids = ["1", tostring(module.grafana_organization_hongbomiao.org_id)]
}

# Data Source
# Data Source - hm-prometheus
module "grafana_data_source_hm_prometheus" {
  source     = "../../../modules/grafana/grafana_data_source"
  for_each   = toset(local.org_ids)
  org_id     = each.value
  name       = "hm-prometheus"
  type       = "prometheus"
  url        = "http://prometheus-kube-prometheus-prometheus.production-hm-prometheus.svc:9090"
  is_default = true
}
# Data Source - hm-loki
module "grafana_data_source_hm_loki" {
  source     = "../../../modules/grafana/grafana_data_source"
  for_each   = toset(local.org_ids)
  org_id     = each.value
  name       = "hm-loki"
  type       = "loki"
  url        = "http://loki-gateway.production-hm-loki.svc:80"
  is_default = false
  http_headers = {
    "X-Scope-OrgID" = "hm"
  }
}

# Dashboard
# Dashboard - Kubernetes
locals {
  dashboard_urls = {
    "k8s-system-api-server" = "https://raw.githubusercontent.com/dotdc/grafana-dashboards-kubernetes/master/dashboards/k8s-system-api-server.json"
    "k8s-system-coredns"    = "https://raw.githubusercontent.com/dotdc/grafana-dashboards-kubernetes/master/dashboards/k8s-system-coredns.json"
    "k8s-views-global"      = "https://raw.githubusercontent.com/dotdc/grafana-dashboards-kubernetes/master/dashboards/k8s-views-global.json"
    "k8s-views-namespaces"  = "https://raw.githubusercontent.com/dotdc/grafana-dashboards-kubernetes/master/dashboards/k8s-views-namespaces.json"
    "k8s-views-nodes"       = "https://raw.githubusercontent.com/dotdc/grafana-dashboards-kubernetes/master/dashboards/k8s-views-nodes.json"
    "k8s-views-pods"        = "https://raw.githubusercontent.com/dotdc/grafana-dashboards-kubernetes/master/dashboards/k8s-views-pods.json"
  }
  org_dashboard_combinations = flatten([
    for org_id in local.org_ids : [
      for dashboard_name, dashboard_url in local.dashboard_urls : {
        key            = "${org_id}-${dashboard_name}"
        org_id         = org_id
        dashboard_name = dashboard_name
        dashboard_url  = dashboard_url
      }
    ]
  ])
}
data "http" "dashboards" {
  for_each = local.dashboard_urls
  url      = each.value
}
module "grafana_folder_kubernetes" {
  source   = "../../../modules/grafana/grafana_folder"
  for_each = toset(local.org_ids)
  org_id   = each.value
  title    = "Kubernetes"
}
module "grafana_dashboard_kubernetes" {
  source      = "../../../modules/grafana/grafana_dashboard"
  for_each    = { for combination in local.org_dashboard_combinations : combination.key => combination }
  org_id      = each.value.org_id
  folder_uid  = module.grafana_folder_kubernetes[each.value.org_id].uid
  config_json = data.http.dashboards[each.value.dashboard_name].response_body
}
