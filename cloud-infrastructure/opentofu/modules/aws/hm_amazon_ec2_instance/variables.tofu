variable "common_tags" {
  type = map(string)
}
variable "instance_name" {
  type = string
}
variable "instance_ami" {
  type = string
}
variable "instance_type" {
  type = string
}
variable "instance_profile" {
  type = string
}
variable "key_name" {
  type = string
}
variable "amazon_vpc_subnet_id" {
  type = string
}
variable "amazon_ec2_security_group_id" {
  type = string
}
