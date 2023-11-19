variable "aws_glue_databrew_job_name" {
  type = string
}
variable "source_name" {
  type = string
}
variable "recipe_version" {
  type = string
}
variable "node_max_number" {
  type = number
}
variable "timeout" {
  type = number
}
variable "input_s3_bucket" {
  type = string
}
variable "input_s3_dir" {
  type = string
}
variable "output_s3_bucket" {
  type = string
}
variable "output_s3_dir" {
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
