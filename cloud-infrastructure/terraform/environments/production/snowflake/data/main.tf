# Department database
locals {
  department_db_department_names = toset([for department in var.department_db_departments : department.name])
  department_db_department_name_schema_name_list = flatten([
    for department in var.department_db_departments :
    [
      for schema in department.schemas :
      {
        department_name = department.name
        schema_name     = schema.name
      }
    ]
  ])
}
module "department_db_database" {
  providers               = { snowflake = snowflake.terraform_role }
  source                  = "../../../../modules/snowflake/hm_snowflake_database"
  for_each                = local.department_db_department_names
  snowflake_database_name = "${var.environment}_${each.value}_DB"
  data_retention_days     = var.database_data_retention_days
}
module "department_db_schema" {
  providers               = { snowflake = snowflake.terraform_role }
  source                  = "../../../../modules/snowflake/hm_snowflake_schema"
  for_each                = { for x in local.department_db_department_name_schema_name_list : "${x.department_name}.${x.schema_name}" => x if x.schema_name != var.public_schema_name }
  snowflake_database_name = "${var.environment}_${each.value.department_name}_DB"
  snowflake_schema_name   = each.value.schema_name
  data_retention_days     = var.database_data_retention_days
  depends_on = [
    module.department_db_database
  ]
}

# hongbomiao Streamlit database
locals {
  hm_streamlit_db_department_names = toset([for department in var.hm_streamlit_db_departments : department.name])
}
module "hm_streamlit_db_database" {
  providers               = { snowflake = snowflake.terraform_role }
  source                  = "../../../../modules/snowflake/hm_snowflake_database"
  snowflake_database_name = "${var.environment}_HM_STREAMLIT_DB"
  data_retention_days     = var.database_data_retention_days
}
module "hm_streamlit_db_department_schema" {
  providers               = { snowflake = snowflake.terraform_role }
  source                  = "../../../../modules/snowflake/hm_snowflake_schema"
  for_each                = local.hm_streamlit_db_department_names
  snowflake_database_name = module.hm_streamlit_db_database.name
  snowflake_schema_name   = each.value
  with_managed_access     = false
  data_retention_days     = var.database_data_retention_days
  depends_on = [
    module.hm_streamlit_db_database
  ]
}

# HM Airbyte database
module "hm_airbyte_db_database" {
  providers               = { snowflake = snowflake.terraform_role }
  source                  = "../../../../modules/snowflake/hm_snowflake_database"
  snowflake_database_name = "${var.environment}_HM_AIRBYTE_DB"
  data_retention_days     = var.database_data_retention_days
}
# HM Kafka database
locals {
  hm_kafka_db_department_names = toset([for department in var.hm_kafka_db_departments : department.name])
}
module "hm_kafka_db_database" {
  providers               = { snowflake = snowflake.terraform_role }
  source                  = "../../../../modules/snowflake/hm_snowflake_database"
  snowflake_database_name = "${var.environment}_HM_KAFKA_DB"
  data_retention_days     = var.database_data_retention_days
}
module "hm_kafka_db_department_schema" {
  providers               = { snowflake = snowflake.terraform_role }
  source                  = "../../../../modules/snowflake/hm_snowflake_schema"
  for_each                = local.hm_kafka_db_department_names
  snowflake_database_name = module.hm_kafka_db_database.name
  snowflake_schema_name   = each.value
  data_retention_days     = var.database_data_retention_days
  depends_on = [
    module.hm_kafka_db_database
  ]
}
