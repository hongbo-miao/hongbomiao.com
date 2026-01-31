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
# Dashboard - Cilium
locals {
  cilium_dashboard_urls = {
    "cilium-dashboard"                   = "https://raw.githubusercontent.com/cilium/cilium/main/install/kubernetes/cilium/files/cilium-agent/dashboards/cilium-dashboard.json"
    "cilium-operator-dashboard"          = "https://raw.githubusercontent.com/cilium/cilium/main/install/kubernetes/cilium/files/cilium-operator/dashboards/cilium-operator-dashboard.json"
    "hubble-dashboard"                   = "https://raw.githubusercontent.com/cilium/cilium/main/install/kubernetes/cilium/files/hubble/dashboards/hubble-dashboard.json"
    "hubble-l7-http-metrics-by-workload" = "https://raw.githubusercontent.com/cilium/cilium/main/install/kubernetes/cilium/files/hubble/dashboards/hubble-l7-http-metrics-by-workload.json"
  }
  cilium_org_dashboard_combinations = flatten([
    for org_id in local.org_ids : [
      for dashboard_name in keys(local.cilium_dashboard_urls) : {
        key            = "${org_id}-${dashboard_name}"
        org_id         = org_id
        dashboard_name = dashboard_name
      }
    ]
  ])
}
data "http" "cilium_dashboards" {
  for_each = local.cilium_dashboard_urls
  url      = each.value
}
module "grafana_folder_cilium" {
  source   = "../../../modules/grafana/grafana_folder"
  for_each = toset(local.org_ids)
  org_id   = each.value
  title    = "Cilium"
}
module "grafana_dashboard_cilium" {
  source      = "../../../modules/grafana/grafana_dashboard"
  for_each    = { for combination in local.cilium_org_dashboard_combinations : combination.key => combination }
  org_id      = each.value.org_id
  folder_uid  = module.grafana_folder_cilium[each.value.org_id].uid
  config_json = data.http.cilium_dashboards[each.value.dashboard_name].response_body
  uid_prefix  = "hm-"
  tags        = [var.managed_tag]
}

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
      for dashboard_name in keys(local.dashboard_urls) : {
        key            = "${org_id}-${dashboard_name}"
        org_id         = org_id
        dashboard_name = dashboard_name
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
  uid_prefix  = "hm-"
  tags        = [var.managed_tag]
}


# Dashboard - NATS Server
# https://github.com/nats-io/prometheus-nats-exporter/blob/main/walkthrough/README.md
# Using non-helm dashboard version because the exporter uses the default "gnatsd_" metrics prefix.
# The "-helm" version expects "nats_" prefix which requires the "-prefix nats" flag on the exporter.
locals {
  nats_org_id = tostring(module.grafana_organization_hongbomiao.org_id)
  # https://github.com/nats-io/prometheus-nats-exporter/pull/393
  nats_server_variable = {
    current     = {}
    definition  = "label_values(gnatsd_varz_cpu, server_id)"
    hide        = 0
    includeAll  = true
    label       = "Server"
    multi       = true
    name        = "server"
    options     = []
    query       = { query = "label_values(gnatsd_varz_cpu, server_id)", refId = "StandardVariableQuery" }
    refresh     = 2
    regex       = ""
    skipUrlSync = false
    sort        = 0
    type        = "query"
  }
}
data "http" "nats_server_dashboard" {
  url = "https://raw.githubusercontent.com/nats-io/prometheus-nats-exporter/main/walkthrough/grafana-nats-dash.json"
}
locals {
  # https://github.com/nats-io/prometheus-nats-exporter/pull/393
  nats_server_dashboard_processed = jsonencode(merge(
    jsondecode(data.http.nats_server_dashboard.response_body),
    {
      templating = {
        list = concat(
          [merge(local.nats_server_variable, {
            datasource = {
              type = "prometheus"
              uid  = module.grafana_data_source_hm_prometheus[local.nats_org_id].uid
            }
          })],
          jsondecode(data.http.nats_server_dashboard.response_body).templating.list
        )
      }
      panels = [
        for panel in jsondecode(data.http.nats_server_dashboard.response_body).panels :
        merge(panel, {
          datasource = {
            type = "prometheus"
            uid  = module.grafana_data_source_hm_prometheus[local.nats_org_id].uid
          }
        })
      ]
    }
  ))
  # https://github.com/nats-io/prometheus-nats-exporter/pull/393
  nats_server_dashboard_with_server_filter = replace(
    local.nats_server_dashboard_processed,
    "/\"expr\":\"(gnatsd_varz_\\w+)\"/",
    "\"expr\":\"$1{server_id=~\\\"$$server\\\"}\""
  )
}
module "grafana_folder_nats" {
  source = "../../../modules/grafana/grafana_folder"
  org_id = local.nats_org_id
  title  = "NATS"
}
module "grafana_dashboard_nats_server" {
  source     = "../../../modules/grafana/grafana_dashboard"
  org_id     = local.nats_org_id
  folder_uid = module.grafana_folder_nats.uid
  config_json = replace(
    local.nats_server_dashboard_with_server_filter,
    "$${DS__NATS-PROMETHEUS}",
    module.grafana_data_source_hm_prometheus[local.nats_org_id].uid
  )
  uid_prefix = "hm-"
  tags       = [var.managed_tag]
}

# Dashboard - NATS JetStream
# https://github.com/nats-io/prometheus-nats-exporter/blob/main/walkthrough/README.md
# Using non-helm dashboard version because the exporter uses the default "gnatsd_" metrics prefix.
data "http" "nats_jetstream_dashboard" {
  url = "https://raw.githubusercontent.com/nats-io/prometheus-nats-exporter/main/walkthrough/grafana-jetstream-dash.json"
}
locals {
  # https://github.com/nats-io/prometheus-nats-exporter/pull/393
  nats_jetstream_dashboard_processed = jsonencode(merge(
    jsondecode(data.http.nats_jetstream_dashboard.response_body),
    {
      templating = {
        list = [
          for variable in jsondecode(data.http.nats_jetstream_dashboard.response_body).templating.list :
          merge(variable, {
            datasource = {
              type = "prometheus"
              uid  = module.grafana_data_source_hm_prometheus[local.nats_org_id].uid
            }
          })
        ]
      }
      panels = [
        for panel in jsondecode(data.http.nats_jetstream_dashboard.response_body).panels :
        merge(panel, {
          datasource = {
            type = "prometheus"
            uid  = module.grafana_data_source_hm_prometheus[local.nats_org_id].uid
          }
        })
      ]
    }
  ))
}
module "grafana_dashboard_nats_jetstream" {
  source      = "../../../modules/grafana/grafana_dashboard"
  org_id      = local.nats_org_id
  folder_uid  = module.grafana_folder_nats.uid
  config_json = local.nats_jetstream_dashboard_processed
  uid_prefix  = "hm-"
  tags        = [var.managed_tag]
}
