terraform {
  required_providers {
    aws = {
      source = "hashicorp/aws"
    }
  }
}

# https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/glue_crawler
# https://docs.aws.amazon.com/glue/latest/dg/crawler-configuration.html
resource "aws_glue_crawler" "main" {
  name          = var.aws_glue_crawler_name
  role          = var.iam_role_arn
  database_name = var.aws_glue_database
  schedule      = var.schedule
  delta_target {
    delta_tables              = var.aws_glue_crawler_delta_tables
    create_native_delta_table = true
    write_manifest            = false
  }
  schema_change_policy {
    delete_behavior = "LOG"
    update_behavior = "LOG"
  }
  lineage_configuration {
    crawler_lineage_settings = "ENABLE"
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
  tags = merge(var.common_tags, {
    "hm:resource_name" = var.aws_glue_crawler_name
  })
}
