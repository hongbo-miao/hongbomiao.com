variable "aws_glue_crawler_name" {
  type = string
}
variable "aws_glue_crawler_delta_tables" {
  type = list(string)
}
variable "aws_glue_database" {
  type = string
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
