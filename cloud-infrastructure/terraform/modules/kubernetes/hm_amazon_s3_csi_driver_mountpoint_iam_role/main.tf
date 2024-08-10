terraform {
  required_providers {
    aws = {
      source = "hashicorp/aws"
    }
  }
}

locals {
  aws_iam_role_name_prefix = "S3CSIDriverMountpointRole"
}
# https://docs.aws.amazon.com/eks/latest/userguide/s3-csi.html
# https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/iam_role
resource "aws_iam_role" "hm_amazon_s3_csi_driver_mountpoint_iam_role" {
  name = "${local.aws_iam_role_name_prefix}-${var.amazon_eks_cluster_name}"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          Federated = var.amazon_eks_cluster_oidc_provider_arn
        }
        Action = "sts:AssumeRoleWithWebIdentity"
        Condition = {
          StringEquals = {
            "${var.amazon_eks_cluster_oidc_provider}:aud" = "sts.amazonaws.com",
            "${var.amazon_eks_cluster_oidc_provider}:sub" = "system:serviceaccount:kube-system:s3-csi-driver-sa"
          }
        }
      }
    ]
  })
  tags = {
    Environment  = var.environment
    Team         = var.team
    ResourceName = "${local.aws_iam_role_name_prefix}-${var.amazon_eks_cluster_name}"
  }
}
# https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/iam_policy
resource "aws_iam_policy" "hm_amazon_s3_csi_driver_mountpoint_iam_policy" {
  name = "${local.aws_iam_role_name_prefix}-${var.amazon_eks_cluster_name}"
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:ListBucket"
        ]
        Resource = [
          "arn:aws:s3:::${var.s3_bucket_name}"
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "s3:AbortMultipartUpload",
          "s3:DeleteObject",
          "s3:GetObject",
          "s3:PutObject"
        ]
        Resource = [
          "arn:aws:s3:::${var.s3_bucket_name}/*"
        ]
      }
    ]
  })
  tags = {
    Environment  = var.environment
    Team         = var.team
    ResourceName = "${local.aws_iam_role_name_prefix}-${var.amazon_eks_cluster_name}"
  }
}
# https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/iam_role_policy_attachment
resource "aws_iam_role_policy_attachment" "hm_amazon_emr_studio_iam_role_policy_attachment" {
  role       = aws_iam_role.hm_amazon_s3_csi_driver_mountpoint_iam_role.name
  policy_arn = aws_iam_policy.hm_amazon_s3_csi_driver_mountpoint_iam_policy.arn
}

