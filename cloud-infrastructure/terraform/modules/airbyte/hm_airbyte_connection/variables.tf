variable "name" {
  type = string
}
variable "source_id" {
  type = string
}
variable "destination_id" {
  type = string
}
variable "streams" {
  type = list(object({
    name      = string
    sync_mode = string
  }))
}
