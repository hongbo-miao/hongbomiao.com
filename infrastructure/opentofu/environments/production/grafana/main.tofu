# Organization
# Organization - hm
module "grafana_organization_hm" {
  source = "../../../modules/grafana/grafana_organization"
  name   = "hm"
}
locals {
  org_ids = [
    "1",
    tostring(module.grafana_organization_hm.org_id),
  ]
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
# Data Source - hm-mimir
module "grafana_data_source_hm_mimir" {
  source     = "../../../modules/grafana/grafana_data_source"
  for_each   = toset(local.org_ids)
  org_id     = each.value
  name       = "hm-mimir"
  type       = "prometheus"
  url        = "http://mimir-gateway.production-hm-mimir.svc:80/prometheus"
  is_default = false
  http_headers = {
    "X-Scope-OrgID" = "hm"
  }
  json_data_encoded = jsonencode({
    prometheusType = "Mimir"
  })
}

# Data Source - hm-tempo
module "grafana_data_source_hm_tempo" {
  source     = "../../../modules/grafana/grafana_data_source"
  for_each   = toset(local.org_ids)
  org_id     = each.value
  name       = "hm-tempo"
  type       = "tempo"
  url        = "http://tempo-gateway.production-hm-tempo.svc:80"
  is_default = false
  http_headers = {
    "X-Scope-OrgID" = "hm"
  }
  # https://grafana.com/docs/grafana/latest/datasources/tempo/configure-tempo-data-source/
  json_data_encoded = jsonencode({
    tracesToLogsV2 = {
      datasourceUid      = module.grafana_data_source_hm_loki[each.value].uid
      spanStartTimeShift = "-1h"
      spanEndTimeShift   = "1h"
      tags               = ["job", "instance", "pod", "namespace"]
      filterByTraceID    = true
      filterBySpanID     = true
      customQuery        = true
      query              = "method=\"$${__span.tags.method}\""
    }
    tracesToMetrics = {
      datasourceUid      = module.grafana_data_source_hm_mimir[each.value].uid
      spanStartTimeShift = "-1h"
      spanEndTimeShift   = "1h"
      tags               = [{ key = "service.name", value = "service" }, { key = "job" }]
      queries = [
        {
          name  = "Sample query"
          query = "sum(rate(traces_spanmetrics_latency_bucket{$$__tags}[5m]))"
        }
      ]
    }
    serviceMap = {
      datasourceUid = module.grafana_data_source_hm_mimir[each.value].uid
    }
    nodeGraph = {
      enabled = true
    }
    search = {
      hide = false
    }
    traceQuery = {
      timeShiftEnabled   = true
      spanStartTimeShift = "-1h"
      spanEndTimeShift   = "1h"
    }
    spanBar = {
      type = "Tag"
      tag  = "http.path"
    }
    streamingEnabled = {
      search = true
    }
  })
}

# Data Source - hm-pyroscope
module "grafana_data_source_hm_pyroscope" {
  source     = "../../../modules/grafana/grafana_data_source"
  for_each   = toset(local.org_ids)
  org_id     = each.value
  name       = "hm-pyroscope"
  type       = "grafana-pyroscope-datasource"
  url        = "http://pyroscope.production-hm-pyroscope.svc:4040"
  is_default = false
  http_headers = {
    "X-Scope-OrgID" = "hm"
  }
}

# Dashboard
# Dashboard - Cilium
module "grafana_folder_cilium" {
  source   = "../../../modules/grafana/grafana_folder"
  for_each = toset(local.org_ids)
  org_id   = each.value
  title    = "Cilium"
}
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
module "grafana_dashboard_cilium" {
  source      = "../../../modules/grafana/grafana_dashboard"
  for_each    = { for combination in local.cilium_org_dashboard_combinations : combination.key => combination }
  org_id      = each.value.org_id
  folder_uid  = module.grafana_folder_cilium[each.value.org_id].uid
  config_json = data.http.cilium_dashboards[each.value.dashboard_name].response_body
  uid_prefix  = var.grafana_dashboard_uid_prefix
  tags        = [var.managed_tag]
}

# Dashboard - Kubernetes
# Dashboard - Kubernetes - Kubernetes
module "grafana_folder_kubernetes" {
  source   = "../../../modules/grafana/grafana_folder"
  for_each = toset(local.org_ids)
  org_id   = each.value
  title    = "Kubernetes"
}
locals {
  kubernetes_dashboard_urls = {
    "k8s-system-api-server" = "https://raw.githubusercontent.com/dotdc/grafana-dashboards-kubernetes/master/dashboards/k8s-system-api-server.json"
    "k8s-system-coredns"    = "https://raw.githubusercontent.com/dotdc/grafana-dashboards-kubernetes/master/dashboards/k8s-system-coredns.json"
    "k8s-views-global"      = "https://raw.githubusercontent.com/dotdc/grafana-dashboards-kubernetes/master/dashboards/k8s-views-global.json"
    "k8s-views-namespaces"  = "https://raw.githubusercontent.com/dotdc/grafana-dashboards-kubernetes/master/dashboards/k8s-views-namespaces.json"
    "k8s-views-nodes"       = "https://raw.githubusercontent.com/dotdc/grafana-dashboards-kubernetes/master/dashboards/k8s-views-nodes.json"
    "k8s-views-pods"        = "https://raw.githubusercontent.com/dotdc/grafana-dashboards-kubernetes/master/dashboards/k8s-views-pods.json"
  }
  kubernetes_org_dashboard_combinations = flatten([
    for org_id in local.org_ids : [
      for dashboard_name in keys(local.kubernetes_dashboard_urls) : {
        key            = "${org_id}-${dashboard_name}"
        org_id         = org_id
        dashboard_name = dashboard_name
      }
    ]
  ])
}
data "http" "kubernetes_dashboards" {
  for_each = local.kubernetes_dashboard_urls
  url      = each.value
}
module "grafana_dashboard_kubernetes" {
  source      = "../../../modules/grafana/grafana_dashboard"
  for_each    = { for combination in local.kubernetes_org_dashboard_combinations : combination.key => combination }
  org_id      = each.value.org_id
  folder_uid  = module.grafana_folder_kubernetes[each.value.org_id].uid
  config_json = data.http.kubernetes_dashboards[each.value.dashboard_name].response_body
  uid_prefix  = var.grafana_dashboard_uid_prefix
  tags        = [var.managed_tag]
}
# Dashboard - Kubernetes - Node Exporter
# https://grafana.com/grafana/dashboards/1860-node-exporter-full/
locals {
  node_exporter_dashboard_urls = {
    "node-exporter-full" = "https://grafana.com/api/dashboards/1860/revisions/latest/download"
  }
  node_exporter_org_dashboard_combinations = flatten([
    for org_id in local.org_ids : [
      for dashboard_name in keys(local.node_exporter_dashboard_urls) : {
        key            = "${org_id}-${dashboard_name}"
        org_id         = org_id
        dashboard_name = dashboard_name
      }
    ]
  ])
}
data "http" "node_exporter_dashboards" {
  for_each = local.node_exporter_dashboard_urls
  url      = each.value
}
module "grafana_dashboard_node_exporter" {
  source      = "../../../modules/grafana/grafana_dashboard"
  for_each    = { for combination in local.node_exporter_org_dashboard_combinations : combination.key => combination }
  org_id      = each.value.org_id
  folder_uid  = module.grafana_folder_kubernetes[each.value.org_id].uid
  config_json = data.http.node_exporter_dashboards[each.value.dashboard_name].response_body
  uid_prefix  = var.grafana_dashboard_uid_prefix
  tags        = [var.managed_tag]
}

# Dashboard - NVIDIA DCGM
# https://grafana.com/grafana/dashboards/12239-nvidia-dcgm-exporter-dashboard/
module "grafana_folder_nvidia_dcgm" {
  source   = "../../../modules/grafana/grafana_folder"
  for_each = toset(local.org_ids)
  org_id   = each.value
  title    = "NVIDIA DCGM"
}
locals {
  nvidia_dcgm_exporter_dashboard_urls = {
    "nvidia-dcgm-exporter" = "https://grafana.com/api/dashboards/12239/revisions/latest/download"
  }
  nvidia_dcgm_exporter_org_dashboard_combinations = flatten([
    for org_id in local.org_ids : [
      for dashboard_name in keys(local.nvidia_dcgm_exporter_dashboard_urls) : {
        key            = "${org_id}-${dashboard_name}"
        org_id         = org_id
        dashboard_name = dashboard_name
      }
    ]
  ])
}
data "http" "nvidia_dcgm_exporter_dashboards" {
  for_each = local.nvidia_dcgm_exporter_dashboard_urls
  url      = each.value
}
# https://github.com/NVIDIA/dcgm-exporter/pull/634
locals {
  nvidia_dcgm_exporter_dashboard_processed = {
    for name, _ in local.nvidia_dcgm_exporter_dashboard_urls : name => replace(
      replace(
        replace(
          data.http.nvidia_dcgm_exporter_dashboards[name].response_body,
          "DCGM_FI_DEV_GPU_UTIL",
          "DCGM_FI_PROF_GR_ENGINE_ACTIVE"
        ),
        "\"format\": \"percent\"",
        "\"format\": \"percentunit\""
      ),
      "\"max\": \"100\"",
      "\"max\": \"1\""
    )
  }
}
module "grafana_dashboard_nvidia_dcgm_exporter" {
  source     = "../../../modules/grafana/grafana_dashboard"
  for_each   = { for combination in local.nvidia_dcgm_exporter_org_dashboard_combinations : combination.key => combination }
  org_id     = each.value.org_id
  folder_uid = module.grafana_folder_nvidia_dcgm[each.value.org_id].uid
  config_json = replace(
    local.nvidia_dcgm_exporter_dashboard_processed[each.value.dashboard_name],
    "$${DS_PROMETHEUS}",
    module.grafana_data_source_hm_prometheus[each.value.org_id].uid
  )
  uid_prefix = var.grafana_dashboard_uid_prefix
  tags       = [var.managed_tag]
}

# Dashboard - NATS
# https://github.com/nats-io/prometheus-nats-exporter/blob/main/walkthrough/README.md
# Using non-helm dashboard version because the exporter uses the default "gnatsd_" metrics prefix.
# The "-helm" version expects "nats_" prefix which requires the "-prefix nats" flag on the exporter.
locals {
  nats_org_id = tostring(module.grafana_organization_hm.org_id)
}
module "grafana_folder_nats" {
  source = "../../../modules/grafana/grafana_folder"
  org_id = local.nats_org_id
  title  = "NATS"
}

# Dashboard - NATS - NATS Server
locals {
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
module "grafana_dashboard_nats_server" {
  source     = "../../../modules/grafana/grafana_dashboard"
  org_id     = local.nats_org_id
  folder_uid = module.grafana_folder_nats.uid
  config_json = replace(
    local.nats_server_dashboard_with_server_filter,
    "$${DS__NATS-PROMETHEUS}",
    module.grafana_data_source_hm_prometheus[local.nats_org_id].uid
  )
  uid_prefix = var.grafana_dashboard_uid_prefix
  tags       = [var.managed_tag]
}

# Dashboard - NATS - NATS JetStream
# https://github.com/nats-io/prometheus-nats-exporter/blob/main/walkthrough/README.md
# Using non-helm dashboard version because the exporter uses the default "gnatsd_" metrics prefix
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
  uid_prefix  = var.grafana_dashboard_uid_prefix
  tags        = [var.managed_tag]
}

# Dashboard - Prefect
# https://github.com/PrefectHQ/prometheus-prefect-exporter/blob/main/dashboards/prefect-flow-run-overview.json
module "grafana_folder_prefect" {
  source   = "../../../modules/grafana/grafana_folder"
  for_each = toset(local.org_ids)
  org_id   = each.value
  title    = "Prefect"
}
locals {
  prefect_dashboard_urls = {
    "prefect-flow-run-overview" = "https://raw.githubusercontent.com/PrefectHQ/prometheus-prefect-exporter/main/dashboards/prefect-flow-run-overview.json"
  }
  prefect_org_dashboard_combinations = flatten([
    for org_id in local.org_ids : [
      for dashboard_name in keys(local.prefect_dashboard_urls) : {
        key            = "${org_id}-${dashboard_name}"
        org_id         = org_id
        dashboard_name = dashboard_name
      }
    ]
  ])
}
data "http" "prefect_dashboards" {
  for_each = local.prefect_dashboard_urls
  url      = each.value
}
module "grafana_dashboard_prefect" {
  source     = "../../../modules/grafana/grafana_dashboard"
  for_each   = { for combination in local.prefect_org_dashboard_combinations : combination.key => combination }
  org_id     = each.value.org_id
  folder_uid = module.grafana_folder_prefect[each.value.org_id].uid
  config_json = replace(
    replace(
      data.http.prefect_dashboards[each.value.dashboard_name].response_body,
      "$${DS_PROMETHEUS}",
      module.grafana_data_source_hm_prometheus[each.value.org_id].uid
    ),
    "$${DS_LOKI}",
    module.grafana_data_source_hm_loki[each.value.org_id].uid
  )
  uid_prefix = var.grafana_dashboard_uid_prefix
  tags       = [var.managed_tag]
}

# Dashboard - Postgres
module "grafana_folder_postgres" {
  source   = "../../../modules/grafana/grafana_folder"
  for_each = toset(local.org_ids)
  org_id   = each.value
  title    = "Postgres"
}
# Dashboard - Postgres - Postgres Overview
# https://github.com/prometheus-community/postgres_exporter/blob/master/postgres_mixin/dashboards/postgres-overview.json
locals {
  postgres_exporter_overview_dashboard_urls = {
    "postgres-overview" = "https://raw.githubusercontent.com/prometheus-community/postgres_exporter/master/postgres_mixin/dashboards/postgres-overview.json"
  }
  postgres_exporter_overview_org_dashboard_combinations = flatten([
    for org_id in local.org_ids : [
      for dashboard_name in keys(local.postgres_exporter_overview_dashboard_urls) : {
        key            = "${org_id}-${dashboard_name}"
        org_id         = org_id
        dashboard_name = dashboard_name
      }
    ]
  ])
}
data "http" "postgres_exporter_overview_dashboards" {
  for_each = local.postgres_exporter_overview_dashboard_urls
  url      = each.value
}
module "grafana_dashboard_postgres_exporter_overview" {
  source      = "../../../modules/grafana/grafana_dashboard"
  for_each    = { for combination in local.postgres_exporter_overview_org_dashboard_combinations : combination.key => combination }
  org_id      = each.value.org_id
  folder_uid  = module.grafana_folder_postgres[each.value.org_id].uid
  config_json = data.http.postgres_exporter_overview_dashboards[each.value.dashboard_name].response_body
  uid_prefix  = var.grafana_dashboard_uid_prefix
  tags        = [var.managed_tag]
}
# Dashboard - Postgres - PostgreSQL Database
# https://grafana.com/grafana/dashboards/9628-postgresql-database/
locals {
  postgres_exporter_database_dashboard_urls = {
    "postgres-database" = "https://grafana.com/api/dashboards/9628/revisions/latest/download"
  }
  postgres_exporter_database_org_dashboard_combinations = flatten([
    for org_id in local.org_ids : [
      for dashboard_name in keys(local.postgres_exporter_database_dashboard_urls) : {
        key            = "${org_id}-${dashboard_name}"
        org_id         = org_id
        dashboard_name = dashboard_name
      }
    ]
  ])
}
data "http" "postgres_exporter_database_dashboards" {
  for_each = local.postgres_exporter_database_dashboard_urls
  url      = each.value
}
module "grafana_dashboard_postgres_exporter_database" {
  source     = "../../../modules/grafana/grafana_dashboard"
  for_each   = { for combination in local.postgres_exporter_database_org_dashboard_combinations : combination.key => combination }
  org_id     = each.value.org_id
  folder_uid = module.grafana_folder_postgres[each.value.org_id].uid
  config_json = replace(
    data.http.postgres_exporter_database_dashboards[each.value.dashboard_name].response_body,
    "$${DS_PROMETHEUS}",
    module.grafana_data_source_hm_prometheus[each.value.org_id].uid
  )
  uid_prefix = var.grafana_dashboard_uid_prefix
  tags       = [var.managed_tag]
}

# Dashboard - vLLM
module "grafana_folder_vllm" {
  source   = "../../../modules/grafana/grafana_folder"
  for_each = toset(local.org_ids)
  org_id   = each.value
  title    = "vLLM"
}
data "http" "vllm_dashboard" {
  url = "https://raw.githubusercontent.com/vllm-project/vllm/main/examples/observability/prometheus_grafana/grafana.json"
}
locals {
  vllm_dashboard_processed = {
    for org_id in local.org_ids : org_id => jsonencode(merge(
      jsondecode(data.http.vllm_dashboard.response_body),
      { uid = "vllm-${org_id}" }
    ))
  }
}
module "grafana_dashboard_vllm" {
  source     = "../../../modules/grafana/grafana_dashboard"
  for_each   = toset(local.org_ids)
  org_id     = each.value
  folder_uid = module.grafana_folder_vllm[each.value].uid
  config_json = replace(
    local.vllm_dashboard_processed[each.value],
    "$${DS_PROMETHEUS}",
    module.grafana_data_source_hm_prometheus[each.value].uid
  )
  uid_prefix = var.grafana_dashboard_uid_prefix
  tags       = [var.managed_tag]
}

# Dashboard - Volcano
module "grafana_folder_volcano" {
  source   = "../../../modules/grafana/grafana_folder"
  for_each = toset(local.org_ids)
  org_id   = each.value
  title    = "Volcano"
}
locals {
  volcano_dashboard_files = {
    "volcano-global-overview-dashboard"    = "${path.module}/files/volcano/volcano-global-overview-dashboard.json"
    "volcano-queue-overview-dashboard"     = "${path.module}/files/volcano/volcano-queue-overview-dashboard.json"
    "volcano-namespace-overview-dashboard" = "${path.module}/files/volcano/volcano-namespace-overview-dashboard.json"
    "volcano-scheduler-internal-dashboard" = "${path.module}/files/volcano/volcano-scheduler-internal-dashboard.json"
    "volcano-scheduler-queue-dashboard"    = "${path.module}/files/volcano/volcano-scheduler-queue-dashboard.json"
  }
  volcano_org_dashboard_combinations = flatten([
    for org_id in local.org_ids : [
      for dashboard_name in keys(local.volcano_dashboard_files) : {
        key            = "${org_id}-${dashboard_name}"
        org_id         = org_id
        dashboard_name = dashboard_name
      }
    ]
  ])
}
module "grafana_dashboard_volcano" {
  source      = "../../../modules/grafana/grafana_dashboard"
  for_each    = { for combination in local.volcano_org_dashboard_combinations : combination.key => combination }
  org_id      = each.value.org_id
  folder_uid  = module.grafana_folder_volcano[each.value.org_id].uid
  config_json = file(local.volcano_dashboard_files[each.value.dashboard_name])
  uid_prefix  = var.grafana_dashboard_uid_prefix
  tags        = [var.managed_tag]
}
