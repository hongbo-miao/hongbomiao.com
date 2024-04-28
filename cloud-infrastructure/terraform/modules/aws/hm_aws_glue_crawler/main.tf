terraform {
  required_providers {
    aws = {
      source = "hashicorp/aws"
    }
  }
}

# https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/glue_crawler
# https://docs.aws.amazon.com/glue/latest/dg/crawler-configuration.html
resource "aws_glue_crawler" "hm_aws_glue_crawler" {
  name          = var.aws_glue_crawler_name
  role          = var.iam_role_arn
  database_name = var.aws_glue_database
  delta_target {
    delta_tables              = var.aws_glue_crawler_delta_tables
    create_native_delta_table = false
    write_manifest            = false
  }
  schema_change_policy {
    delete_behavior = "LOG"
  }
  configuration = jsonencode(
    {
      Version = 1.0,
      CrawlerOutput = {
        Partitions = { AddOrUpdateBehavior = "InheritFromTable" },
        Tables     = { AddOrUpdateBehavior = "MergeNewColumns" }
      }
      Grouping = {
        TableGroupingPolicy = "CombineCompatibleSchemas"
      }
      CreatePartitionIndex = true
    }
  )
  tags = {
    Environment  = var.environment
    Team         = var.team
    ResourceName = var.aws_glue_crawler_name
  }
}
