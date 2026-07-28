variable "zone_id" {
  type = string
}
variable "name" {
  type = string
}
variable "ttl" {
  type = number
  # 1 = automatic, per docs
  default = 1
}
variable "type" {
  type    = string
  default = "A"
}
variable "content" {
  type    = string
  default = null
}
variable "tags" {
  type = list(string)
}
