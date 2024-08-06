terraform {
  required_providers {
    aws = {
      source = "hashicorp/aws"
    }
  }
}

data "aws_caller_identity" "main" {}

# https://docs.aws.amazon.com/elasticloadbalancing/latest/network/load-balancer-access-logs.html
# https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/s3_bucket_policy
resource "aws_s3_bucket_policy" "hm_aws_network_load_balancer_s3_bucket_policy" {
  bucket = var.s3_bucket_name
  policy = jsonencode({
    Version = "2012-10-17",
    Id      = "AWSLogDeliveryWrite",
    Statement = [
      {
        Sid    = "AWSLogDeliveryAclCheck",
        Effect = "Allow",
        Principal = {
          Service = "delivery.logs.amazonaws.com"
        },
        Action   = "s3:GetBucketAcl",
        Resource = "arn:aws:s3:::${var.s3_bucket_name}",
        Condition = {
          StringEquals = {
            "aws:SourceAccount" = [data.aws_caller_identity.main.account_id]
          },
          ArnLike = {
            "aws:SourceArn" = ["arn:aws:logs:us-west-2:${data.aws_caller_identity.main.account_id}:*"]
          }
        }
      },
      {
        Sid    = "AWSLogDeliveryWrite",
        Effect = "Allow",
        Principal = {
          Service = "delivery.logs.amazonaws.com"
        },
        Action   = "s3:PutObject",
        Resource = "arn:aws:s3:::${var.s3_bucket_name}/AWSLogs/*",
        Condition = {
          StringEquals = {
            "s3:x-amz-acl"      = "bucket-owner-full-control",
            "aws:SourceAccount" = [data.aws_caller_identity.main.account_id]
          },
          ArnLike = {
            "aws:SourceArn" = ["arn:aws:logs:us-west-2:${data.aws_caller_identity.main.account_id}:*"]
          }
        }
      }
    ]
  })
}
