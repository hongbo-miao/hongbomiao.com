terraform {
  required_providers {
    aws = {
      source = "hashicorp/aws"
    }
  }
}

# https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/route53_record
resource "aws_route53_record" "hm_amazon_route_53_record" {
  zone_id = "xxxxxxxxxxxxxxxxxxxxx"
  name    = var.amazon_route_53_record_name
  records = var.amazon_route_53_record_values
  type    = "A"
  ttl     = 180
}
