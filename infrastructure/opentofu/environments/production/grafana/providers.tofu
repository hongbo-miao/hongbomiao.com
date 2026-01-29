provider "grafana" {
  url  = "https://grafana.hongbomiao.com"
  auth = "${var.grafana_admin_username}:${var.grafana_admin_password}"
}
