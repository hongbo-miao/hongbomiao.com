module "cloudflare_dns_record_dmarc" {
  source  = "../../../modules/cloudflare/dns_record"
  zone_id = var.cloudflare_zone_id
  name    = "_dmarc.${var.cloudflare_domain}"
  type    = "TXT"
  content = "\"v=DMARC1; p=reject; rua=mailto:9ccc0584efb340fea2b93f69bad0ae4f@dmarc-reports.cloudflare.net\""
  tags    = var.common_tags
}

module "cloudflare_dns_record_spf" {
  source  = "../../../modules/cloudflare/dns_record"
  zone_id = var.cloudflare_zone_id
  name    = var.cloudflare_domain
  type    = "TXT"
  content = "\"v=spf1 -all\""
  tags    = var.common_tags
}
