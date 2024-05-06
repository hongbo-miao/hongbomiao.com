variable "environment" {
  type = string
}
variable "snowflake_public_schema_name" {
  type = string
}
variable "production_database_data_retention_days" {
  type = number
}
variable "production_department_db_departments" {
  type = list(object({
    name = string
    schemas = list(object({
      name = string
    }))
  }))
}
variable "production_hm_kafka_db_departments" {
  type = list(object({
    name = string
  }))
}
