output "uri" {
  value = "s3://${aws_s3_object.hm_amazon_s3_object.bucket}/${aws_s3_object.hm_amazon_s3_object.key}"
}
output "s3_key" {
  value = var.s3_key
}
