variable "environment" {
  type = string
}
variable "snowflake_public_schema_name" {
  type = string
}
variable "snowflake_sysadmin" {
  type = string
}
variable "hongbomiao_departments" {
  type = list(object({
    name             = string
    admin_user_names = list(string)
    schemas = list(object({
      name                  = string
      read_only_user_names  = list(string)
      read_write_user_names = list(string)
    }))
  }))
}
variable "warehouse_auto_suspend_min" {
  type = number
}
variable "database_data_retention_days" {
  type = number
}
variable "development_hm_kafka_db_product_read_write_user_rsa_public_key_without_header_and_trailer" {
  type = string
}
