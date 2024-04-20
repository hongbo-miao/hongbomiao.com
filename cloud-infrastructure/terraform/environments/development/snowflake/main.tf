terraform {
  backend "s3" {
    region = "us-west-2"
    bucket = "hm-terraform-bucket"
    key    = "development/snowflake/terraform.tfstate"
  }
  required_providers {
    # https://registry.terraform.io/providers/Snowflake-Labs/snowflake/latest
    snowflake = {
      source  = "Snowflake-Labs/snowflake"
      version = "0.89.0"
    }
  }
  # terraform version
  required_version = ">= 1.7"
}

# Snowflake
provider "snowflake" {
  alias = "hm_development_terraform_read_write_role"
  role  = "HM_DEVELOPMENT_TERRAFORM_READ_WRITE_ROLE"
}

locals {
  department_names = toset([for department in var.hongbomiao_departments : department.name])

  department_name_schema_name_list = flatten([
    for department in var.hongbomiao_departments :
    [
      for schema in department.schemas :
      {
        department_name  = department.name
        admin_user_names = department.admin_user_names
        schema_name      = schema.name
      }
    ]
  ])

  department_name_admin_user_name_list = flatten([
    for department in var.hongbomiao_departments :
    [
      for admin_user_name in department.admin_user_names :
      {
        department_name = department.name
        admin_user_name = admin_user_name
      }
    ]
  ])

  department_name_schema_name_read_only_user_name_list = flatten([
    for department in var.hongbomiao_departments :
    [
      for schema in department.schemas :
      [
        for read_only_user_name in schema.read_only_user_names :
        {
          department_name     = department.name
          schema_name         = schema.name
          read_only_user_name = read_only_user_name
        }
      ]
    ]
  ])

  department_name_schema_name_read_write_user_name_list = flatten([
    for department in var.hongbomiao_departments :
    [
      for schema in department.schemas :
      [
        for read_write_user_name in schema.read_write_user_names :
        {
          department_name      = department.name
          schema_name          = schema.name
          read_write_user_name = read_write_user_name
        }
      ]
    ]
  ])
}

# General warehouse
module "snowflake_development_hm_general_wh_warehouse" {
  providers                = { snowflake = snowflake.hm_development_terraform_read_write_role }
  source                   = "../../../modules/snowflake/hm_snowflake_warehouse"
  snowflake_warehouse_name = "HM_DEVELOPMENT_HM_GENERAL_WH"
  snowflake_warehouse_size = "xsmall"
  auto_suspend_min         = var.warehouse_auto_suspend_min
}
# Department database
module "snowflake_development_department_db_database" {
  providers               = { snowflake = snowflake.hm_development_terraform_read_write_role }
  source                  = "../../../modules/snowflake/hm_snowflake_database"
  for_each                = local.department_names
  snowflake_database_name = "DEVELOPMENT_DEPARTMENT_${each.value}_DB"
  data_retention_days     = var.database_data_retention_days
}
module "snowflake_development_department_db_schema" {
  providers               = { snowflake = snowflake.hm_development_terraform_read_write_role }
  source                  = "../../../modules/snowflake/hm_snowflake_schema"
  for_each                = { for x in local.department_name_schema_name_list : "${x.department_name}.${x.schema_name}" => x if x.schema_name != var.snowflake_public_schema_name }
  snowflake_database_name = "DEVELOPMENT_DEPARTMENT_${each.value.department_name}_DB"
  snowflake_schema_name   = each.value.schema_name
  depends_on = [
    module.snowflake_development_department_db_database
  ]
}
# Read only role
module "snowflake_development_department_db_schema_read_only_role" {
  providers           = { snowflake = snowflake.hm_development_terraform_read_write_role }
  source              = "../../../modules/snowflake/hm_snowflake_role"
  for_each            = { for x in local.department_name_schema_name_list : "${x.department_name}.${x.schema_name}" => x }
  snowflake_role_name = "HM_DEVELOPMENT_DEPARTMENT_${each.value.department_name}_DB_${each.value.schema_name}_READ_ONLY_ROLE"
}
module "snowflake_grant_database_privileges_to_development_department_db_schema_read_only_role" {
  providers               = { snowflake = snowflake.hm_development_terraform_read_write_role }
  source                  = "../../../modules/snowflake/hm_snowflake_grant_database_privileges_to_role"
  for_each                = { for x in local.department_name_schema_name_list : "${x.department_name}.${x.schema_name}" => x }
  snowflake_role_name     = "HM_DEVELOPMENT_DEPARTMENT_${each.value.department_name}_DB_${each.value.schema_name}_READ_ONLY_ROLE"
  privileges              = ["USAGE"]
  snowflake_database_name = "DEVELOPMENT_DEPARTMENT_${each.value.department_name}_DB"
  depends_on = [
    module.snowflake_development_department_db_schema_read_only_role,
    module.snowflake_development_department_db_database,
    module.snowflake_development_department_db_schema
  ]
}
module "snowflake_grant_schema_privileges_to_development_department_db_schema_read_only_role" {
  providers               = { snowflake = snowflake.hm_development_terraform_read_write_role }
  source                  = "../../../modules/snowflake/hm_snowflake_grant_schema_privileges_to_role"
  for_each                = { for x in local.department_name_schema_name_list : "${x.department_name}.${x.schema_name}" => x }
  snowflake_role_name     = "HM_DEVELOPMENT_DEPARTMENT_${each.value.department_name}_DB_${each.value.schema_name}_READ_ONLY_ROLE"
  privileges              = ["USAGE"]
  snowflake_database_name = "DEVELOPMENT_DEPARTMENT_${each.value.department_name}_DB"
  snowflake_schema_name   = each.value.schema_name
  depends_on = [
    module.snowflake_development_department_db_schema_read_only_role,
    module.snowflake_development_department_db_database,
    module.snowflake_development_department_db_schema
  ]
}
module "snowflake_grant_all_future_table_in_schema_privileges_to_development_department_db_schema_read_only_role" {
  providers               = { snowflake = snowflake.hm_development_terraform_read_write_role }
  source                  = "../../../modules/snowflake/hm_snowflake_grant_future_table_in_schema_privileges_to_role"
  for_each                = { for x in local.department_name_schema_name_list : "${x.department_name}.${x.schema_name}" => x }
  snowflake_role_name     = "HM_DEVELOPMENT_DEPARTMENT_${each.value.department_name}_DB_${each.value.schema_name}_READ_ONLY_ROLE"
  privileges              = ["SELECT"]
  snowflake_database_name = "DEVELOPMENT_DEPARTMENT_${each.value.department_name}_DB"
  snowflake_schema_name   = each.value.schema_name
  depends_on = [
    module.snowflake_development_department_db_schema_read_only_role,
    module.snowflake_development_department_db_database,
    module.snowflake_development_department_db_schema
  ]
}
module "snowflake_grant_warehouse_privileges_to_development_department_db_schema_read_only_role" {
  providers                = { snowflake = snowflake.hm_development_terraform_read_write_role }
  source                   = "../../../modules/snowflake/hm_snowflake_grant_warehouse_privileges_to_role"
  for_each                 = { for x in local.department_name_schema_name_list : "${x.department_name}.${x.schema_name}" => x }
  snowflake_role_name      = "HM_DEVELOPMENT_DEPARTMENT_${each.value.department_name}_DB_${each.value.schema_name}_READ_ONLY_ROLE"
  privileges               = ["USAGE"]
  snowflake_warehouse_name = module.snowflake_development_hm_general_wh_warehouse.name
  depends_on = [
    module.snowflake_development_department_db_schema_read_only_role,
    module.snowflake_development_hm_general_wh_warehouse
  ]
}
module "snowflake_grant_development_department_db_schema_read_only_role_to_development_department_db_schema_read_only_user" {
  providers           = { snowflake = snowflake.hm_development_terraform_read_write_role }
  source              = "../../../modules/snowflake/hm_snowflake_grant_role_to_user"
  for_each            = { for x in local.department_name_schema_name_read_only_user_name_list : "${x.department_name}.${x.schema_name}.${x.read_only_user_name}" => x }
  snowflake_role_name = "HM_DEVELOPMENT_DEPARTMENT_${each.value.department_name}_DB_${each.value.schema_name}_READ_ONLY_ROLE"
  snowflake_user_name = each.value.read_only_user_name
  depends_on = [
    module.snowflake_grant_warehouse_privileges_to_development_department_db_schema_read_only_role
  ]
}
# Read write role
module "snowflake_development_department_db_schema_read_write_role" {
  providers           = { snowflake = snowflake.hm_development_terraform_read_write_role }
  source              = "../../../modules/snowflake/hm_snowflake_role"
  for_each            = { for x in local.department_name_schema_name_list : "${x.department_name}.${x.schema_name}" => x }
  snowflake_role_name = "HM_DEVELOPMENT_DEPARTMENT_${each.value.department_name}_DB_${each.value.schema_name}_READ_WRITE_ROLE"
}
module "snowflake_grant_schema_privileges_to_development_department_db_schema_read_write_role" {
  providers               = { snowflake = snowflake.hm_development_terraform_read_write_role }
  source                  = "../../../modules/snowflake/hm_snowflake_grant_schema_privileges_to_role"
  for_each                = { for x in local.department_name_schema_name_list : "${x.department_name}.${x.schema_name}" => x }
  snowflake_role_name     = "HM_DEVELOPMENT_DEPARTMENT_${each.value.department_name}_DB_${each.value.schema_name}_READ_WRITE_ROLE"
  privileges              = ["CREATE TABLE", "CREATE VIEW"]
  snowflake_database_name = "DEVELOPMENT_DEPARTMENT_${each.value.department_name}_DB"
  snowflake_schema_name   = each.value.schema_name
  depends_on = [
    module.snowflake_development_department_db_schema_read_write_role,
    module.snowflake_development_department_db_database,
    module.snowflake_development_department_db_schema
  ]
}
module "snowflake_grant_all_future_table_in_schema_privileges_to_development_department_db_schema_read_write_role" {
  providers               = { snowflake = snowflake.hm_development_terraform_read_write_role }
  source                  = "../../../modules/snowflake/hm_snowflake_grant_future_table_in_schema_privileges_to_role"
  for_each                = { for x in local.department_name_schema_name_list : "${x.department_name}.${x.schema_name}" => x }
  snowflake_role_name     = "HM_DEVELOPMENT_DEPARTMENT_${each.value.department_name}_DB_${each.value.schema_name}_READ_WRITE_ROLE"
  privileges              = ["INSERT", "UPDATE", "DELETE"]
  snowflake_database_name = "DEVELOPMENT_DEPARTMENT_${each.value.department_name}_DB"
  snowflake_schema_name   = each.value.schema_name
  depends_on = [
    module.snowflake_development_department_db_schema_read_write_role,
    module.snowflake_development_department_db_database,
    module.snowflake_development_department_db_schema
  ]
}
module "snowflake_grant_development_department_db_schema_read_only_role_to_development_department_db_schema_read_write_role" {
  providers                  = { snowflake = snowflake.hm_development_terraform_read_write_role }
  source                     = "../../../modules/snowflake/hm_snowflake_grant_role_to_role"
  for_each                   = { for x in local.department_name_schema_name_list : "${x.department_name}.${x.schema_name}" => x }
  snowflake_role_name        = "HM_DEVELOPMENT_DEPARTMENT_${each.value.department_name}_DB_${each.value.schema_name}_READ_ONLY_ROLE"
  snowflake_parent_role_name = "HM_DEVELOPMENT_DEPARTMENT_${each.value.department_name}_DB_${each.value.schema_name}_READ_WRITE_ROLE"
  depends_on = [
    module.snowflake_development_department_db_schema_read_only_role,
    module.snowflake_grant_database_privileges_to_development_department_db_schema_read_only_role,
    module.snowflake_grant_all_future_table_in_schema_privileges_to_development_department_db_schema_read_only_role,
    module.snowflake_grant_warehouse_privileges_to_development_department_db_schema_read_only_role,
    module.snowflake_development_department_db_schema_read_write_role
  ]
}
module "snowflake_grant_development_department_db_schema_read_write_role_to_development_department_db_schema_read_write_user" {
  providers           = { snowflake = snowflake.hm_development_terraform_read_write_role }
  source              = "../../../modules/snowflake/hm_snowflake_grant_role_to_user"
  for_each            = { for x in local.department_name_schema_name_read_write_user_name_list : "${x.department_name}.${x.schema_name}.${x.read_write_user_name}" => x }
  snowflake_role_name = "HM_DEVELOPMENT_DEPARTMENT_${each.value.department_name}_DB_${each.value.schema_name}_READ_WRITE_ROLE"
  snowflake_user_name = each.value.read_write_user_name
  depends_on = [
    module.snowflake_development_department_db_schema_read_only_role
  ]
}
# Admin role
module "snowflake_development_department_db_admin_role" {
  providers           = { snowflake = snowflake.hm_development_terraform_read_write_role }
  source              = "../../../modules/snowflake/hm_snowflake_role"
  for_each            = local.department_names
  snowflake_role_name = "HM_DEVELOPMENT_DEPARTMENT_${each.value}_DB_ADMIN_ROLE"
}
module "snowflake_grant_database_all_privileges_to_role_to_development_department_db_admin_role" {
  providers               = { snowflake = snowflake.hm_development_terraform_read_write_role }
  source                  = "../../../modules/snowflake/hm_snowflake_grant_database_all_privileges_to_role"
  for_each                = local.department_names
  snowflake_role_name     = "HM_DEVELOPMENT_DEPARTMENT_${each.value}_DB_ADMIN_ROLE"
  snowflake_database_name = "DEVELOPMENT_DEPARTMENT_${each.value}_DB"
  depends_on = [
    module.snowflake_development_department_db_admin_role,
    module.snowflake_development_department_db_database
  ]
}
module "snowflake_grant_existing_schema_all_privileges_to_role_to_development_department_db_admin_role" {
  providers               = { snowflake = snowflake.hm_development_terraform_read_write_role }
  source                  = "../../../modules/snowflake/hm_snowflake_grant_existing_schema_all_privileges_to_role"
  for_each                = local.department_names
  snowflake_role_name     = "HM_DEVELOPMENT_DEPARTMENT_${each.value}_DB_ADMIN_ROLE"
  snowflake_database_name = "DEVELOPMENT_DEPARTMENT_${each.value}_DB"
  depends_on = [
    module.snowflake_development_department_db_admin_role,
    module.snowflake_development_department_db_database
  ]
}
module "snowflake_grant_future_schema_all_privileges_to_role_to_development_department_db_admin_role" {
  providers               = { snowflake = snowflake.hm_development_terraform_read_write_role }
  source                  = "../../../modules/snowflake/hm_snowflake_grant_future_schema_all_privileges_to_role"
  for_each                = local.department_names
  snowflake_role_name     = "HM_DEVELOPMENT_DEPARTMENT_${each.value}_DB_ADMIN_ROLE"
  snowflake_database_name = "DEVELOPMENT_DEPARTMENT_${each.value}_DB"
  depends_on = [
    module.snowflake_development_department_db_admin_role,
    module.snowflake_development_department_db_database
  ]
}
module "snowflake_grant_existing_table_all_privileges_to_role_to_development_department_db_admin_role" {
  providers               = { snowflake = snowflake.hm_development_terraform_read_write_role }
  source                  = "../../../modules/snowflake/hm_snowflake_grant_existing_table_all_privileges_to_role"
  for_each                = local.department_names
  snowflake_role_name     = "HM_DEVELOPMENT_DEPARTMENT_${each.value}_DB_ADMIN_ROLE"
  snowflake_database_name = "DEVELOPMENT_DEPARTMENT_${each.value}_DB"
  depends_on = [
    module.snowflake_development_department_db_admin_role,
    module.snowflake_development_department_db_database
  ]
}
module "snowflake_grant_future_table_all_privileges_to_role_to_development_department_db_admin_role" {
  providers               = { snowflake = snowflake.hm_development_terraform_read_write_role }
  source                  = "../../../modules/snowflake/hm_snowflake_grant_future_table_all_privileges_to_role"
  for_each                = local.department_names
  snowflake_role_name     = "HM_DEVELOPMENT_DEPARTMENT_${each.value}_DB_ADMIN_ROLE"
  snowflake_database_name = "DEVELOPMENT_DEPARTMENT_${each.value}_DB"
  depends_on = [
    module.snowflake_development_department_db_admin_role,
    module.snowflake_development_department_db_database
  ]
}
module "snowflake_transfer_development_department_database_ownership_to_sysadmin_role" {
  providers               = { snowflake = snowflake.hm_development_terraform_read_write_role }
  source                  = "../../../modules/snowflake/hm_snowflake_transfer_database_ownership_to_role"
  for_each                = local.department_names
  snowflake_role_name     = var.snowflake_sysadmin
  snowflake_database_name = "DEVELOPMENT_DEPARTMENT_${each.value}_DB"
  depends_on = [
    module.snowflake_development_department_db_admin_role,
    module.snowflake_development_department_db_database
  ]
}
module "snowflake_grant_development_department_db_schema_read_write_role_to_development_department_db_admin_role" {
  providers                  = { snowflake = snowflake.hm_development_terraform_read_write_role }
  source                     = "../../../modules/snowflake/hm_snowflake_grant_role_to_role"
  for_each                   = { for x in local.department_name_schema_name_list : "${x.department_name}.${x.schema_name}" => x }
  snowflake_role_name        = "HM_DEVELOPMENT_DEPARTMENT_${each.value.department_name}_DB_${each.value.schema_name}_READ_WRITE_ROLE"
  snowflake_parent_role_name = "HM_DEVELOPMENT_DEPARTMENT_${each.value.department_name}_DB_ADMIN_ROLE"
  depends_on = [
    module.snowflake_development_department_db_schema_read_write_role,
    module.snowflake_grant_development_department_db_schema_read_only_role_to_development_department_db_schema_read_write_role,
    module.snowflake_grant_schema_privileges_to_development_department_db_schema_read_write_role,
    module.snowflake_grant_all_future_table_in_schema_privileges_to_development_department_db_schema_read_write_role,
    module.snowflake_development_department_db_admin_role
  ]
}
module "snowflake_transfer_development_department_db_schema_ownership_to_sysadmin_role" {
  providers               = { snowflake = snowflake.hm_development_terraform_read_write_role }
  source                  = "../../../modules/snowflake/hm_snowflake_transfer_schema_ownership_to_role"
  for_each                = { for x in local.department_name_schema_name_list : "${x.department_name}.${x.schema_name}" => x }
  snowflake_role_name     = var.snowflake_sysadmin
  snowflake_database_name = "DEVELOPMENT_DEPARTMENT_${each.value.department_name}_DB"
  snowflake_schema_name   = each.value.schema_name
  depends_on = [
    module.snowflake_development_department_db_admin_role,
    module.snowflake_development_department_db_database,
    module.snowflake_development_department_db_schema
  ]
}
module "snowflake_grant_development_department_db_schema_admin_role_to_development_department_db_schema_admin_user" {
  providers           = { snowflake = snowflake.hm_development_terraform_read_write_role }
  source              = "../../../modules/snowflake/hm_snowflake_grant_role_to_user"
  for_each            = { for x in local.department_name_admin_user_name_list : "${x.department_name}.${x.admin_user_name}" => x }
  snowflake_role_name = "HM_DEVELOPMENT_DEPARTMENT_${each.value.department_name}_DB_ADMIN_ROLE"
  snowflake_user_name = each.value.admin_user_name
  depends_on = [
    module.snowflake_development_department_db_admin_role
  ]
}

# Streamlit
module "snowflake_development_hongbomiao_streamlit_wh_warehouse" {
  providers                = { snowflake = snowflake.hm_development_terraform_read_write_role }
  source                   = "../../../modules/snowflake/hm_snowflake_warehouse"
  snowflake_warehouse_name = "HM_DEVELOPMENT_HONGBOMIAO_STREAMLIT_WH"
  snowflake_warehouse_size = "xsmall"
  auto_suspend_min         = var.warehouse_auto_suspend_min
}
module "snowflake_development_hongbomiao_streamlit_db_database" {
  providers               = { snowflake = snowflake.hm_development_terraform_read_write_role }
  source                  = "../../../modules/snowflake/hm_snowflake_database"
  snowflake_database_name = "DEVELOPMENT_HONGBOMIAO_STREAMLIT_DB"
  data_retention_days     = var.database_data_retention_days
}
module "snowflake_development_hongbomiao_streamlit_db_department_schema" {
  providers               = { snowflake = snowflake.hm_development_terraform_read_write_role }
  source                  = "../../../modules/snowflake/hm_snowflake_schema"
  for_each                = local.department_names
  snowflake_database_name = module.snowflake_development_hongbomiao_streamlit_db_database.name
  snowflake_schema_name   = each.value
  depends_on = [
    module.snowflake_development_hongbomiao_streamlit_db_database
  ]
}
module "snowflake_development_hongbomiao_streamlit_db_department_read_write_role" {
  providers           = { snowflake = snowflake.hm_development_terraform_read_write_role }
  source              = "../../../modules/snowflake/hm_snowflake_role"
  for_each            = local.department_names
  snowflake_role_name = "HM_${module.snowflake_development_hongbomiao_streamlit_db_database.name}_${each.value}_READ_WRITE_ROLE"
}
module "snowflake_grant_database_privileges_to_development_hongbomiao_streamlit_db_read_write_role" {
  providers               = { snowflake = snowflake.hm_development_terraform_read_write_role }
  source                  = "../../../modules/snowflake/hm_snowflake_grant_database_privileges_to_role"
  for_each                = { for x in local.department_name_schema_name_list : "${x.department_name}.${x.schema_name}" => x }
  snowflake_role_name     = "HM_${module.snowflake_development_hongbomiao_streamlit_db_database.name}_${each.value.department_name}_READ_WRITE_ROLE"
  privileges              = ["USAGE"]
  snowflake_database_name = module.snowflake_development_hongbomiao_streamlit_db_database.name
  depends_on = [
    module.snowflake_development_hongbomiao_streamlit_db_department_read_write_role,
    module.snowflake_development_hongbomiao_streamlit_db_database
  ]
}
module "snowflake_grant_schema_privileges_to_development_hongbomiao_streamlit_db_read_write_role" {
  providers               = { snowflake = snowflake.hm_development_terraform_read_write_role }
  source                  = "../../../modules/snowflake/hm_snowflake_grant_schema_privileges_to_role"
  for_each                = { for x in local.department_name_schema_name_list : "${x.department_name}.${x.schema_name}" => x }
  snowflake_role_name     = "HM_${module.snowflake_development_hongbomiao_streamlit_db_database.name}_${each.value.department_name}_READ_WRITE_ROLE"
  privileges              = ["USAGE", "CREATE STREAMLIT", "CREATE STAGE"]
  snowflake_database_name = module.snowflake_development_hongbomiao_streamlit_db_database.name
  snowflake_schema_name   = each.value.department_name
  depends_on = [
    module.snowflake_development_hongbomiao_streamlit_db_department_read_write_role,
    module.snowflake_development_hongbomiao_streamlit_db_database,
    module.snowflake_development_hongbomiao_streamlit_db_department_schema
  ]
}
module "snowflake_grant_warehouse_privileges_to_development_department_db_department_read_write_role" {
  providers                = { snowflake = snowflake.hm_development_terraform_read_write_role }
  source                   = "../../../modules/snowflake/hm_snowflake_grant_warehouse_privileges_to_role"
  for_each                 = { for x in local.department_name_schema_name_list : "${x.department_name}.${x.schema_name}" => x }
  snowflake_role_name      = "HM_${module.snowflake_development_hongbomiao_streamlit_db_database.name}_${each.value.department_name}_READ_WRITE_ROLE"
  privileges               = ["USAGE"]
  snowflake_warehouse_name = module.snowflake_development_hongbomiao_streamlit_wh_warehouse.name
  depends_on = [
    module.snowflake_development_hongbomiao_streamlit_db_department_read_write_role,
    module.snowflake_development_hongbomiao_streamlit_wh_warehouse
  ]
}
# Empty department role to help share Streamlit app to different department end users
module "snowflake_development_streamlit_app_for_department_end_users_role" {
  providers           = { snowflake = snowflake.hm_development_terraform_read_write_role }
  source              = "../../../modules/snowflake/hm_snowflake_role"
  for_each            = local.department_names
  snowflake_role_name = "HM_DEVELOPMENT_STREAMLIT_APP_FOR_DEPARTMENT_${each.value}_END_USERS_ROLE"
}

# Kafka
module "snowflake_development_hm_kafka_wh_warehouse" {
  providers                = { snowflake = snowflake.hm_development_terraform_read_write_role }
  source                   = "../../../modules/snowflake/hm_snowflake_warehouse"
  snowflake_warehouse_name = "HM_DEVELOPMENT_HM_KAFKA_WH"
  snowflake_warehouse_size = "xsmall"
  auto_suspend_min         = var.warehouse_auto_suspend_min
}
module "snowflake_development_hm_kafka_db_database" {
  providers               = { snowflake = snowflake.hm_development_terraform_read_write_role }
  source                  = "../../../modules/snowflake/hm_snowflake_database"
  snowflake_database_name = "DEVELOPMENT_HM_KAFKA_DB"
  data_retention_days     = var.database_data_retention_days
}
module "snowflake_development_hm_kafka_db_product_schema" {
  providers               = { snowflake = snowflake.hm_development_terraform_read_write_role }
  source                  = "../../../modules/snowflake/hm_snowflake_schema"
  snowflake_database_name = module.snowflake_development_hm_kafka_db_database.name
  snowflake_schema_name   = "PRODUCT"
  depends_on = [
    module.snowflake_development_hm_kafka_db_database
  ]
}
module "snowflake_development_hm_kafka_db_product_read_write_role" {
  providers           = { snowflake = snowflake.hm_development_terraform_read_write_role }
  source              = "../../../modules/snowflake/hm_snowflake_role"
  snowflake_role_name = "HM_${module.snowflake_development_hm_kafka_db_database.name}_${module.snowflake_development_hm_kafka_db_product_schema.name}_READ_WRITE_ROLE"
}
module "snowflake_development_hm_kafka_db_product_read_write_user" {
  providers                                 = { snowflake = snowflake.hm_development_terraform_read_write_role }
  source                                    = "../../../modules/snowflake/hm_snowflake_user"
  snowflake_user_name                       = "HM_${module.snowflake_development_hm_kafka_db_database.name}_${module.snowflake_development_hm_kafka_db_product_schema.name}_READ_WRITE_USER"
  default_role                              = module.snowflake_development_hm_kafka_db_product_read_write_role.name
  rsa_public_key_without_header_and_trailer = var.development_hm_kafka_db_product_read_write_user_rsa_public_key_without_header_and_trailer
  depends_on = [
    module.snowflake_development_hm_kafka_db_product_read_write_role
  ]
}
module "snowflake_grant_development_hm_kafka_db_product_read_write_role_to_development_hm_kafka_db_product_read_write_user" {
  providers           = { snowflake = snowflake.hm_development_terraform_read_write_role }
  source              = "../../../modules/snowflake/hm_snowflake_grant_role_to_user"
  snowflake_role_name = module.snowflake_development_hm_kafka_db_product_read_write_role.name
  snowflake_user_name = module.snowflake_development_hm_kafka_db_product_read_write_user.name
  depends_on = [
    module.snowflake_development_hm_kafka_db_product_read_write_user,
    module.snowflake_development_hm_kafka_db_product_read_write_role
  ]
}
# https://docs.snowflake.com/en/user-guide/kafka-connector-install
module "snowflake_grant_database_privileges_to_development_hm_kafka_db_product_read_write_role" {
  providers               = { snowflake = snowflake.hm_development_terraform_read_write_role }
  source                  = "../../../modules/snowflake/hm_snowflake_grant_database_privileges_to_role"
  snowflake_role_name     = module.snowflake_development_hm_kafka_db_product_read_write_role.name
  privileges              = ["USAGE"]
  snowflake_database_name = module.snowflake_development_hm_kafka_db_database.name
  depends_on = [
    module.snowflake_development_hm_kafka_db_product_read_write_role,
    module.snowflake_development_hm_kafka_db_database
  ]
}
module "snowflake_grant_schema_privileges_to_development_hm_kafka_db_product_read_write_role" {
  providers               = { snowflake = snowflake.hm_development_terraform_read_write_role }
  source                  = "../../../modules/snowflake/hm_snowflake_grant_schema_privileges_to_role"
  snowflake_role_name     = module.snowflake_development_hm_kafka_db_product_read_write_role.name
  privileges              = ["USAGE", "CREATE TABLE", "CREATE STAGE", "CREATE PIPE"]
  snowflake_database_name = module.snowflake_development_hm_kafka_db_database.name
  snowflake_schema_name   = module.snowflake_development_hm_kafka_db_product_schema.name
  depends_on = [
    module.snowflake_development_hm_kafka_db_product_read_write_role,
    module.snowflake_development_hm_kafka_db_database,
    module.snowflake_development_hm_kafka_db_product_schema
  ]
}
module "snowflake_grant_warehouse_privileges_to_development_hm_kafka_db_product_read_write_role" {
  providers                = { snowflake = snowflake.hm_development_terraform_read_write_role }
  source                   = "../../../modules/snowflake/hm_snowflake_grant_warehouse_privileges_to_role"
  snowflake_role_name      = module.snowflake_development_hm_kafka_db_product_read_write_role.name
  privileges               = ["USAGE"]
  snowflake_warehouse_name = module.snowflake_development_hm_kafka_wh_warehouse.name
  depends_on = [
    module.snowflake_development_hm_kafka_db_product_read_write_role,
    module.snowflake_development_hm_kafka_wh_warehouse
  ]
}
