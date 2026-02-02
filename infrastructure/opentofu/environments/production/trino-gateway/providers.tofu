# https://registry.terraform.io/providers/paragor/trinogateway/latest/docs
provider "trinogateway" {
  endpoint = "http://trino-gateway.hongbomiao.com"
  login    = var.trino_gateway_user_name
  password = var.trino_gateway_password
}
