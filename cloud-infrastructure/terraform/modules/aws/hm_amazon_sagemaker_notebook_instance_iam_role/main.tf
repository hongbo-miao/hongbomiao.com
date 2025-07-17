terraform {
  required_providers {
    aws = {
      source = "hashicorp/aws"
    }
  }
}

# https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/iam_role
resource "aws_iam_role" "sagemaker_notebook_instance_role" {
  name = "SageMakerExecutionRole-${var.amazon_sagemaker_notebook_instance_name}"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          Service = "sagemaker.amazonaws.com"
        }
        Action = "sts:AssumeRole"
      }
    ]
  })
  tags = {
    Environment  = var.environment
    Team         = var.team
    ResourceName = "AmazonSageMakerExecutionRole-${var.amazon_sagemaker_notebook_instance_name}"
  }
}
# https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/iam_role_policy
resource "aws_iam_role_policy" "sagemaker_notebook_instance_role_s3_policy" {
  name = "SageMakerExecutionPolicyForS3-${var.amazon_sagemaker_notebook_instance_name}"
  role = aws_iam_role.sagemaker_notebook_instance_role.name
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:ListBucket"
        ]
        Resource = [
          "arn:aws:s3:::*"
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "s3:DeleteObject",
          "s3:GetObject",
          "s3:PutObject"
        ]
        Resource = [
          "arn:aws:s3:::*"
        ]
      }
    ]
  })
}
# https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/iam_role_policy_attachment
resource "aws_iam_role_policy_attachment" "sagemaker_notebook_instance_role_policy_attachment_amazon_sagemaker_canvas_ai_services_access" {
  role       = aws_iam_role.sagemaker_notebook_instance_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSageMakerCanvasAIServicesAccess"
}
resource "aws_iam_role_policy_attachment" "sagemaker_notebook_instance_role_policy_attachment_amazon_sagemaker_canvas_full_access" {
  role       = aws_iam_role.sagemaker_notebook_instance_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSageMakerCanvasFullAccess"
}
resource "aws_iam_role_policy_attachment" "sagemaker_notebook_instance_role_policy_attachment_amazon_sagemaker_full_access" {
  role       = aws_iam_role.sagemaker_notebook_instance_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess"
}
