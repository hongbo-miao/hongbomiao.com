output "uri" {
  value = "s3://${aws_s3_object.main.bucket}/${aws_s3_object.main.key}"
}
output "s3_key" {
  value = var.s3_key
}
