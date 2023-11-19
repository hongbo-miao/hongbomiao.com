# https://registry.terraform.io/providers/hashicorp/awscc/latest/docs/resources/databrew_dataset
resource "awscc_databrew_dataset" "hm_aws_databrew_dataset" {
  name = var.aws_glue_databrew_dataset_name
  input = {
    s3_input_definition = {
      bucket = var.input_s3_bucket
      key    = var.input_s3_dir
    }
  }
  format_options = {
    json = {
      multi_line = true
    }
  }
  tags = [
    {
      key   = "Environment"
      value = var.environment
    },
    {
      key   = "Team"
      value = var.team
    },
    {
      key   = "Name"
      value = var.aws_glue_databrew_dataset_name
    }
  ]
}
