module "backend_trino" {
  source = "../../../modules/trino-gateway/trino_gateway_backend"
  name   = "trino"
  # https://trinodb.github.io/trino-gateway/routing-rules/
  # By default, requests are routed based on the `X-Trino-Routing-Group` header, or to the default routing group (adhoc) if the header is absent
  routing_group = "adhoc"
  proxy_to      = "http://trino.production-hm-trino.svc:8080"
  external_url  = "https://trino.hongbomiao.com"
  active        = true
}
