variable "aws_batch_compute_environment_name" {
  type = string
}
variable "amazon_ec2_security_group_ids" {
  type = list(string)
}
variable "amazon_vpc_subnet_ids" {
  type = list(string)
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
