variable "common_tags" {
  type = map(string)
}
variable "aws_glue_crawler_name" {
  type = string
}
variable "aws_glue_crawler_delta_tables" {
  type = list(string)
}
variable "aws_glue_database" {
  type = string
}
variable "schedule" {
  type = string
}
variable "iam_role_arn" {
  type = string
}
