variable "org_id" {
  type = string
}
variable "name" {
  type = string
}
variable "type" {
  type = string
}
variable "url" {
  type = string
}
variable "is_default" {
  type = bool
}
variable "http_headers" {
  type      = map(string)
  default   = {}
  sensitive = true
}
variable "json_data_encoded" {
  type    = string
  default = null
}
variable "secure_json_data_encoded" {
  type      = string
  default   = null
  sensitive = true
}
