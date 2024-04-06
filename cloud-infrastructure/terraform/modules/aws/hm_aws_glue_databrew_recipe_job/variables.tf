variable "aws_glue_databrew_recipe_job_name" {
  type = string
}
variable "iam_role_arn" {
  type = string
}
variable "aws_glue_databrew_dataset_name" {
  type = string
}
variable "recipe_name" {
  type = string
}
variable "recipe_version" {
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
variable "output_max_file_number" {
  type = number
}
variable "environment" {
  type = string
}
variable "team" {
  type = string
}
