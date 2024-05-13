# Amazon S3 bucket - hm-development-bucket
module "development_hm_development_bucket_amazon_s3_bucket" {
  providers      = { aws = aws.development }
  source         = "../../../../modules/aws/hm_amazon_s3_bucket"
  s3_bucket_name = "hm-development-bucket"
  environment    = var.environment
  team           = var.team
}
