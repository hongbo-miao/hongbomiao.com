variable "common_tags" {
  type = map(string)
}
variable "aws_glue_job_name" {
  type = string
}
variable "aws_glue_version" {
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
variable "spark_conf" {
  type = string
}
variable "timeout_min" {
  type = number
}
variable "iam_role_arn" {
  type = string
}
