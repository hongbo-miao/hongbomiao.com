# https://registry.terraform.io/providers/hashicorp/awscc/latest/docs/resources/databrew_job
resource "awscc_databrew_job" "hm_aws_glue_databrew_job" {
  name         = var.aws_glue_databrew_job_name
  role_arn     = var.aws_iam_role
  type         = "RECIPE"
  dataset_name = "adsb-2x-flight-trace-data"
  max_capacity = 10
  recipe = {
    name    = var.recipe_name
    version = var.recipe_version
  }
  outputs = [
    {
      location = {
        bucket = var.output_s3_bucket
        key    = var.output_s3_dir
      }
      format    = "PARQUET"
      overwrite = true
    }
  ]
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
      value = var.aws_glue_databrew_job_name
    }
  ]
}
