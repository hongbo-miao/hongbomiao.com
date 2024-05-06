variable "environment" {
  type = string
}
variable "snowflake_public_schema_name" {
  type = string
}
variable "development_warehouse_auto_suspend_min" {
  type = number
}
variable "development_department_db_departments" {
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
variable "development_hm_kafka_db_departments" {
  type = list(object({
    name                                                      = string
    read_write_user_rsa_public_key_without_header_and_trailer = string
  }))
}
