variable "name" {
  type = string
}
variable "workspace_id" {
  type = string
}
variable "postgres_host" {
  type = string
}
variable "postgres_port" {
  type = number
}
variable "postgres_user_name" {
  type = string
}
variable "postgres_password" {
  type = string
}
variable "postgres_database" {
  type = string
}
variable "postgres_schemas" {
  type = list(string)
}
variable "postgres_replication_slot" {
  type = string
}
variable "postgres_publication" {
  type = string
}

