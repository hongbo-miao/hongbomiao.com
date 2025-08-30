variable "common_tags" {
  type = map(string)
}
variable "aws_glue_databrew_job_nickname" {
  type = string
}
variable "input_s3_bucket_name" {
  type = string
}
variable "output_s3_bucket_name" {
  type = string
}
