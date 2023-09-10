# https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/glue_job
# https://docs.aws.amazon.com/glue/latest/dg/aws-glue-programming-etl-glue-arguments.html
resource "aws_glue_job" "hm_aws_glue_job" {
  name              = var.aws_glue_job_name
  role_arn          = var.aws_iam_role
  glue_version      = "4.0"
  worker_type       = "G.1X"
  number_of_workers = 10
  timeout           = 30
  command {
    script_location = var.spark_script_s3_location
    python_version  = 3
  }
  execution_property {
    max_concurrent_runs = 1
  }
  default_arguments = {
    "--job-language"                     = "python"
    "--job-bookmark-option"              = "job-bookmark-enable"
    "--enable-glue-datacatalog"          = true
    "--enable-auto-scaling"              = true
    "--enable-spark-ui"                  = true
    "--spark-event-logs-path"            = "s3://aws-glue-assets-272394222652-us-west-2/sparkHistoryLogs/"
    "--enable-metrics"                   = true
    "--enable-continuous-cloudwatch-log" = true
    "--TempDir"                          = "s3://aws-glue-assets-272394222652-us-west-2/temporary/"
    "--datalake-formats"                 = "delta"
  }
  tags = {
    Environment = var.environment
    Team        = var.team
    Name        = var.aws_glue_job_name
  }
}
