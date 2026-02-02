terraform {
  required_providers {
    aws = {
      source = "registry.opentofu.org/hashicorp/aws"
    }
  }
}

# https://search.opentofu.org/provider/hashicorp/aws/latest/docs/resources/msk_scram_secret_association
resource "aws_msk_scram_secret_association" "main" {
  cluster_arn     = var.amazon_msk_cluster_arn
  secret_arn_list = [var.aws_secrets_manager_secret_arn]
}
