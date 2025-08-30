variable "snowflake_schema_name" {
  type = string
}
variable "snowflake_database_name" {
  type = string
}
variable "data_retention_days" {
  type = number
}
variable "with_managed_access" {
  type    = bool
  default = true
}
