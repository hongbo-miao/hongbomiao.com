variable "common_tags" {
  type = map(string)
}
variable "aws_batch_job_definition_name" {
  type = string
}
variable "iam_role_arn" {
  type = string
}
