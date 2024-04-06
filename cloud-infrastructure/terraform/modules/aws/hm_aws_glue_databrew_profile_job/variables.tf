variable "aws_glue_databrew_profile_job_name" {
  type = string
}
variable "iam_role_arn" {
  type = string
}
variable "aws_glue_databrew_dataset_name" {
  type = string
}
variable "spark_worker_max_number" {
  type = number
}
variable "timeout_min" {
  type = number
}
variable "output_s3_bucket_name" {
  type = string
}
variable "output_s3_key" {
  type = string
}
variable "environment" {
  type = string
}
variable "team" {
  type = string
}
