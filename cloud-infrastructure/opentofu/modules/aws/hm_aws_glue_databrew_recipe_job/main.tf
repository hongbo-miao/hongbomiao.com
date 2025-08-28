terraform {
  required_providers {
    awscc = {
      source = "hashicorp/awscc"
    }
  }
}

# https://registry.terraform.io/providers/hashicorp/awscc/latest/docs/resources/databrew_job
resource "awscc_databrew_job" "glue_databrew_recipe_job" {
  name         = var.aws_glue_databrew_recipe_job_name
  role_arn     = var.iam_role_arn
  type         = "RECIPE"
  dataset_name = var.aws_glue_databrew_dataset_name
  max_capacity = var.spark_worker_max_number
  timeout      = var.timeout_min
  recipe = {
    name    = var.recipe_name
    version = var.recipe_version
  }
  outputs = [
    {
      location = {
        bucket = var.output_s3_bucket_name
        key    = var.output_s3_key
      }
      format           = "PARQUET"
      max_output_files = var.output_max_file_number
      overwrite        = true
    }
  ]
  tags = concat(
    [for key, value in var.common_tags : {
      key   = key
      value = value
    }],
    [
      {
        key   = "hm:resource_name"
        value = var.aws_glue_databrew_recipe_job_name
      }
    ]
  )
}
