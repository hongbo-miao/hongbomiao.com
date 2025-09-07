variable "environment" {
  type = string
}
variable "public_schema_name" {
  type = string
}
variable "department_warehouse_auto_suspend_min" {
  type = number
}
variable "airbyte_warehouse_auto_suspend_min" {
  type = number
}
variable "kafka_warehouse_auto_suspend_min" {
  type = number
}
variable "streamlit_warehouse_auto_suspend_min" {
  type = number
}
variable "department_db_departments" {
  type = list(object({
    name             = string
    admin_user_names = optional(list(string), [])
    schemas = list(object({
      name                  = string
      read_only_user_names  = optional(list(string), [])
      read_write_user_names = optional(list(string), [])
      # RSA public key without header and trailer
      read_only_service_account_rsa_public_key = optional(string, null)
      # RSA public key without header and trailer
      read_write_service_account_rsa_public_key = optional(string, null)
    }))
  }))
}
variable "hm_streamlit_db_departments" {
  type = list(object({
    name          = string
    creator_names = optional(list(string), [])
    user_names    = optional(list(string), [])
  }))
}
variable "hm_kafka_db_departments" {
  type = list(object({
    name = string
    # RSA public key without header and trailer
    read_write_service_account_rsa_public_key = string
  }))
}
# RSA public key without header and trailer
variable "hm_airbyte_db_owner_service_account_rsa_public_key" {
  type = string
}
