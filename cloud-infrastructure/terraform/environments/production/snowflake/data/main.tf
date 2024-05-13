# Department database
locals {
  production_department_db_department_names = toset([for department in var.production_department_db_departments : department.name])
  production_department_db_department_name_schema_name_list = flatten([
    for department in var.production_department_db_departments :
    [
      for schema in department.schemas :
      {
        department_name = department.name
        schema_name     = schema.name
      }
    ]
  ])
}
module "snowflake_production_department_db_database" {
  providers               = { snowflake = snowflake.hm_production_terraform_read_write_role }
  source                  = "../../../../modules/snowflake/hm_snowflake_database"
  for_each                = local.production_department_db_department_names
  snowflake_database_name = "DEVELOPMENT_DEPARTMENT_${each.value}_DB"
  data_retention_days     = var.production_database_data_retention_days
}
module "snowflake_production_department_db_schema" {
  providers               = { snowflake = snowflake.hm_production_terraform_read_write_role }
  source                  = "../../../../modules/snowflake/hm_snowflake_schema"
  for_each                = { for x in local.production_department_db_department_name_schema_name_list : "${x.department_name}.${x.schema_name}" => x if x.schema_name != var.snowflake_public_schema_name }
  snowflake_database_name = "DEVELOPMENT_DEPARTMENT_${each.value.department_name}_DB"
  snowflake_schema_name   = each.value.schema_name
  depends_on = [
    module.snowflake_production_department_db_database
  ]
}

# hongbomiao Streamlit database
module "snowflake_production_hongbomiao_streamlit_db_database" {
  providers               = { snowflake = snowflake.hm_production_terraform_read_write_role }
  source                  = "../../../../modules/snowflake/hm_snowflake_database"
  snowflake_database_name = "DEVELOPMENT_HONGBOMIAO_STREAMLIT_DB"
  data_retention_days     = var.production_database_data_retention_days
}
module "snowflake_production_hongbomiao_streamlit_db_department_schema" {
  providers               = { snowflake = snowflake.hm_production_terraform_read_write_role }
  source                  = "../../../../modules/snowflake/hm_snowflake_schema"
  for_each                = local.production_department_db_department_names
  snowflake_database_name = module.snowflake_production_hongbomiao_streamlit_db_database.name
  snowflake_schema_name   = each.value
  depends_on = [
    module.snowflake_production_hongbomiao_streamlit_db_database
  ]
}

# HM Kafka database
locals {
  production_hm_kafka_db_department_names = toset([for department in var.production_hm_kafka_db_departments : department.name])
}
module "snowflake_production_hm_kafka_db_database" {
  providers               = { snowflake = snowflake.hm_production_terraform_read_write_role }
  source                  = "../../../../modules/snowflake/hm_snowflake_database"
  snowflake_database_name = "DEVELOPMENT_HM_KAFKA_DB"
  data_retention_days     = var.production_database_data_retention_days
}
module "snowflake_production_hm_kafka_db_department_schema" {
  providers               = { snowflake = snowflake.hm_production_terraform_read_write_role }
  source                  = "../../../../modules/snowflake/hm_snowflake_schema"
  for_each                = local.production_hm_kafka_db_department_names
  snowflake_database_name = module.snowflake_production_hm_kafka_db_database.name
  snowflake_schema_name   = each.value
  depends_on = [
    module.snowflake_production_hm_kafka_db_database
  ]
}
