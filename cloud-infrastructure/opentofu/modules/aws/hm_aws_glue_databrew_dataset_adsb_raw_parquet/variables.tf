variable "common_tags" {
  type = map(string)
}
variable "aws_glue_databrew_dataset_name" {
  type = string
}
variable "input_s3_bucket_name" {
  type = string
}
variable "input_s3_key" {
  type = string
}
