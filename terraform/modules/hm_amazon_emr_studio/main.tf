# https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/iam_role
# https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/iam_role_policy
# https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/iam_role_policy_attachment
resource "aws_iam_role" "hm_amazon_emr_studio_iam_role" {
  name = "AmazonEMRStudioServiceRole-${var.amazon_emr_studio_name}"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = "sts:AssumeRole"
        Principal = {
          Service = "elasticmapreduce.amazonaws.com"
        }
      }
    ]
  })
  tags = {
    Environment = var.environment
    Team        = var.team
    Name        = "AmazonEMRStudioServiceRole-${var.amazon_emr_studio_name}"
  }
}
resource "aws_iam_role_policy" "hm_amazon_emr_studio_iam_role_s3_policy" {
  name = "AmazonEMRStudioServicePolicyForS3-${var.amazon_emr_studio_name}"
  role = aws_iam_role.hm_amazon_emr_studio_iam_role.name
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:DeleteObject",
          "s3:GetEncryptionConfiguration",
          "s3:GetObject",
          "s3:ListBucket",
          "s3:PutObject"
        ]
        Resource = [
          "arn:aws:s3:::${var.s3_bucket}",
          "arn:aws:s3:::${var.s3_bucket}/*"
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "s3:ListAllMyBuckets"
        ]
        Resource = [
          "*"
        ]
      }
    ]
  })
}
resource "aws_iam_role_policy_attachment" "hm_amazon_emr_studio_iam_role_policy_attachment" {
  role       = aws_iam_role.hm_amazon_emr_studio_iam_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonElasticMapReduceEditorsRole"
}

# https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/emr_studio
resource "aws_emr_studio" "hm_amazon_emr_studio" {
  name                        = var.amazon_emr_studio_name
  auth_mode                   = "IAM"
  default_s3_location         = var.s3_uri
  service_role                = aws_iam_role.hm_amazon_emr_studio_iam_role.arn
  engine_security_group_id    = "sg-xxxxxxxxxxxxxxxxx"
  workspace_security_group_id = "sg-xxxxxxxxxxxxxxxxx"
  vpc_id                      = "vpc-xxxxxxxxxxxxxxxxx"
  subnet_ids                  = ["subnet-xxxxxxxxxxxxxxxxx"]
  tags = {
    Environment = var.environment
    Team        = var.team
    Name        = var.amazon_emr_studio_name
  }
}
