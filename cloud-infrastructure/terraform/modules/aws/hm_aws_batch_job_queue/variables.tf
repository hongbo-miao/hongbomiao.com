variable "aws_batch_job_queue_name" {
  type = string
}
variable "aws_batch_compute_environment_arns" {
  type = list(string)
}
variable "environment" {
  type = string
}
variable "team" {
  type = string
}
