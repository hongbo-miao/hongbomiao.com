terraform {
  required_providers {
    aws = {
      source = "registry.opentofu.org/hashicorp/aws"
    }
  }
}

# https://search.opentofu.org/provider/hashicorp/aws/latest/docs/resources/route53_record
resource "aws_route53_record" "main" {
  zone_id = "xxxxxxxxxxxxxxxxxxxxx"
  name    = var.amazon_route_53_record_name
  records = var.amazon_route_53_record_values
  type    = "A"
  ttl     = 180
}
