variable "common_tags" {
  type = map(string)
}
variable "amazon_sagemaker_notebook_instance_name" {
  type = string
}
variable "instance_type" {
  type = string
}
variable "iam_role_arn" {
  type = string
}
