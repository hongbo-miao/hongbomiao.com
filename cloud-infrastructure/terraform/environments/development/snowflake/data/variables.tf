variable "environment" {
  type = string
}
variable "snowflake_public_schema_name" {
  type = string
}
variable "development_database_data_retention_days" {
  type = number
}
variable "development_department_db_departments" {
  type = list(object({
    name = string
    schemas = list(object({
      name = string
    }))
  }))
}
variable "development_hm_kafka_db_departments" {
  type = list(object({
    name = string
  }))
}
