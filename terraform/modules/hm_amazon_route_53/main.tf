# https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/route53_record
resource "aws_route53_record" "hm_amazon_route_53_record" {
  zone_id = "xxxxxxxxxxxxxxxxxxxxx"
  name    = var.name
  records = var.records
  type    = "A"
  ttl     = 180
}
