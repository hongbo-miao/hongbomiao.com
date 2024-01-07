variable "aws_glue_job_name" {
  type = string
}
variable "spark_script_s3_uri" {
  type = string
}
variable "spark_worker_type" {
  type = string
}
variable "spark_worker_max_number" {
  type = number
}
variable "timeout_min" {
  type = number
}
variable "iam_role_arn" {
  type = string
}
variable "environment" {
  type = string
}
variable "team" {
  type = string
}
