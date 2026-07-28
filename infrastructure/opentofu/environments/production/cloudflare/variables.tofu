variable "environment" {
  type = string
}
variable "cloudflare_domain" {
  type = string
}
variable "cloudflare_zone_id" {
  type = string
}
variable "common_tags" {
  type = list(string)
}
