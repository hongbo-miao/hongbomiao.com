terraform {
  backend "s3" {
    region = "us-west-2"
    bucket = "hm-terraform-bucket"
    key    = "development/snowflake/data/terraform.tfstate"
  }
  required_providers {
    # https://registry.terraform.io/providers/Snowflake-Labs/snowflake/latest
    snowflake = {
      source  = "Snowflake-Labs/snowflake"
      version = "0.90.0"
    }
  }
  required_version = ">= 1.7"
}

provider "snowflake" {
  alias = "hm_development_terraform_read_write_role"
  role  = "HM_DEVELOPMENT_TERRAFORM_READ_WRITE_ROLE"
}


# Department database
locals {
  development_department_db_department_names = toset([for department in var.development_department_db_departments : department.name])
  development_department_db_department_name_schema_name_list = flatten([
    for department in var.development_department_db_departments :
    [
      for schema in department.schemas :
      {
        department_name = department.name
        schema_name     = schema.name
      }
    ]
  ])
}
module "snowflake_development_department_db_database" {
  providers               = { snowflake = snowflake.hm_development_terraform_read_write_role }
  source                  = "../../../../modules/snowflake/hm_snowflake_database"
  for_each                = local.development_department_db_department_names
  snowflake_database_name = "DEVELOPMENT_DEPARTMENT_${each.value}_DB"
  data_retention_days     = var.development_database_data_retention_days
}
module "snowflake_development_department_db_schema" {
  providers               = { snowflake = snowflake.hm_development_terraform_read_write_role }
  source                  = "../../../../modules/snowflake/hm_snowflake_schema"
  for_each                = { for x in local.development_department_db_department_name_schema_name_list : "${x.department_name}.${x.schema_name}" => x if x.schema_name != var.snowflake_public_schema_name }
  snowflake_database_name = "DEVELOPMENT_DEPARTMENT_${each.value.department_name}_DB"
  snowflake_schema_name   = each.value.schema_name
  depends_on = [
    module.snowflake_development_department_db_database
  ]
}

# hongbomiao Streamlit database
module "snowflake_development_hongbomiao_streamlit_db_database" {
  providers               = { snowflake = snowflake.hm_development_terraform_read_write_role }
  source                  = "../../../../modules/snowflake/hm_snowflake_database"
  snowflake_database_name = "DEVELOPMENT_HONGBOMIAO_STREAMLIT_DB"
  data_retention_days     = var.development_database_data_retention_days
}
module "snowflake_development_hongbomiao_streamlit_db_department_schema" {
  providers               = { snowflake = snowflake.hm_development_terraform_read_write_role }
  source                  = "../../../../modules/snowflake/hm_snowflake_schema"
  for_each                = local.development_department_db_department_names
  snowflake_database_name = module.snowflake_development_hongbomiao_streamlit_db_database.name
  snowflake_schema_name   = each.value
  depends_on = [
    module.snowflake_development_hongbomiao_streamlit_db_database
  ]
}

# HM Kafka database
locals {
  development_hm_kafka_db_department_names = toset([for department in var.development_hm_kafka_db_departments : department.name])
}
module "snowflake_development_hm_kafka_db_database" {
  providers               = { snowflake = snowflake.hm_development_terraform_read_write_role }
  source                  = "../../../../modules/snowflake/hm_snowflake_database"
  snowflake_database_name = "DEVELOPMENT_HM_KAFKA_DB"
  data_retention_days     = var.development_database_data_retention_days
}
module "snowflake_development_hm_kafka_db_department_schema" {
  providers               = { snowflake = snowflake.hm_development_terraform_read_write_role }
  source                  = "../../../../modules/snowflake/hm_snowflake_schema"
  for_each                = local.development_hm_kafka_db_department_names
  snowflake_database_name = module.snowflake_development_hm_kafka_db_database.name
  snowflake_schema_name   = each.value
  depends_on = [
    module.snowflake_development_hm_kafka_db_database
  ]
}
