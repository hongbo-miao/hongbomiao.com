terraform {
  required_providers {
    cloudflare = {
      source = "cloudflare/cloudflare"
    }
  }
}

# https://search.opentofu.org/provider/cloudflare/cloudflare/latest/docs/resources/dns_record
resource "cloudflare_dns_record" "main" {
  zone_id = var.zone_id
  name    = var.name
  ttl     = var.ttl
  type    = var.type
  content = var.content
  tags    = var.tags
}
