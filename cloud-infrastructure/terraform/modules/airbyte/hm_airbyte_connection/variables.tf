variable "source_id" {
  type = string
}
variable "destination_id" {
  type = string
}
variable "destination_name" {
  type = string
}
variable "streams" {
  type = list(object({
    name      = string
    sync_mode = string
  }))
}
variable "schedule_type" {
  type = string
}
variable "schedule_cron_expression" {
  type    = string
  default = null
}
