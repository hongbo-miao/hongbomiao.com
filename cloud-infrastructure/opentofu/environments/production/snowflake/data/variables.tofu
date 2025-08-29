variable "environment" {
  type = string
}
variable "public_schema_name" {
  type = string
}
variable "database_data_retention_days" {
  type = number
}
variable "department_db_departments" {
  type = list(object({
    name = string
    schemas = list(object({
      name = string
    }))
  }))
}
variable "hm_streamlit_db_departments" {
  type = list(object({
    name = string
  }))
}
variable "hm_kafka_db_departments" {
  type = list(object({
    name = string
  }))
}
