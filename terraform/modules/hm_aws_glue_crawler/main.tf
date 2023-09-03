# https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/glue_crawler
resource "aws_glue_crawler" "itg_aws_glue_crawler" {
  database_name = "hm_delta_db"
  name          = var.aws_glue_crawler_name
  role          = var.aws_glue_crawler_role
  delta_target {
    delta_tables              = var.aws_glue_crawler_delta_tables
    create_native_delta_table = true
    write_manifest            = false
  }
  tags = {
    Environment = var.environment
    Team        = var.team
    Name        = var.aws_glue_crawler_name
  }
}
