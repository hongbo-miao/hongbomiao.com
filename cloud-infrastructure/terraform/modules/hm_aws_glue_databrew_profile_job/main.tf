terraform {
  required_providers {
    awscc = {
      source = "hashicorp/awscc"
    }
  }
}

# https://registry.terraform.io/providers/hashicorp/awscc/latest/docs/resources/databrew_job
resource "awscc_databrew_job" "hm_aws_glue_databrew_profile_job" {
  name         = var.aws_glue_databrew_profile_job_name
  role_arn     = var.iam_role_arn
  type         = "PROFILE"
  dataset_name = var.aws_glue_databrew_dataset_name
  max_capacity = var.spark_worker_max_number
  timeout      = var.timeout_min
  job_sample = {
    mode = "CUSTOM_ROWS"
    size = 10000
  }
  output_location = {
    bucket = var.output_s3_bucket
    key    = var.output_s3_dir
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
      value = var.aws_glue_databrew_profile_job_name
    }
  ]
}
