variable "amazon_emr_studio_name" {
  type = string
}
variable "s3_bucket" {
  type = string
}
variable "s3_uri" {
  type = string
}
variable "engine_security_group_id" {
  type = string
}
variable "subnet_ids" {
  type = list(string)
}
variable "vpc_id" {
  type = string
}
variable "workspace_security_group_id" {
  type = string
}
variable "environment" {
  type = string
}
variable "team" {
  type = string
}
