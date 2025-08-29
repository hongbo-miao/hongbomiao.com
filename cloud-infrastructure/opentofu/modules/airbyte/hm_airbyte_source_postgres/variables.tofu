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
variable "postgres_schema" {
  type = string
}
variable "initial_waiting_time_s" {
  type = number
}
variable "tunnel_method" {
  type = object({
    no_tunnel = optional(map(any))
    ssh_key_authentication = optional(object({
      tunnel_host = string
      tunnel_port = number
      tunnel_user = string
      ssh_key     = string
    }))
  })
  default = {
    no_tunnel = {}
  }
}
