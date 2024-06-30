data "terraform_remote_state" "hm_terraform_remote_state_production_snowflake_data" {
  backend = "s3"
  config = {
    region = "us-west-2"
    bucket = "hm-terraform-hongbomiao"
    key    = "production/snowflake/data/terraform.tfstate"
  }
}

# Department warehouse
module "snowflake_production_hm_department_wh_warehouse" {
  providers                = { snowflake = snowflake.hm_production_terraform_read_write_role }
  source                   = "../../../../modules/snowflake/hm_snowflake_warehouse"
  snowflake_warehouse_name = "HM_PRODUCTION_HM_DEPARTMENT_WH"
  snowflake_warehouse_size = "xsmall"
  auto_suspend_min         = var.production_department_warehouse_auto_suspend_min
}
# Department role
locals {
  production_department_db_department_names = toset([for department in var.production_department_db_departments : department.name])
  production_department_db_department_name_schema_name_list = flatten([
    for department in var.production_department_db_departments :
    [
      for schema in department.schemas :
      {
        department_name  = department.name
        admin_user_names = department.admin_user_names
        schema_name      = schema.name
      }
    ]
  ])
  production_department_db_department_name_admin_user_name_list = flatten([
    for department in var.production_department_db_departments :
    [
      for admin_user_name in department.admin_user_names :
      {
        department_name = department.name
        admin_user_name = admin_user_name
      }
    ]
  ])
  production_department_db_department_name_schema_name_read_only_user_name_list = flatten([
    for department in var.production_department_db_departments :
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
  production_department_db_department_name_schema_name_read_write_user_name_list = flatten([
    for department in var.production_department_db_departments :
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
# Department role - read only role
module "snowflake_production_department_db_schema_read_only_role" {
  providers           = { snowflake = snowflake.hm_production_terraform_read_write_role }
  source              = "../../../../modules/snowflake/hm_snowflake_role"
  for_each            = { for x in local.production_department_db_department_name_schema_name_list : "${x.department_name}.${x.schema_name}" => x }
  snowflake_role_name = "HM_PRODUCTION_DEPARTMENT_${each.value.department_name}_DB_${each.value.schema_name}_READ_ONLY_ROLE"
}
module "snowflake_grant_database_privileges_to_production_department_db_schema_read_only_role" {
  providers               = { snowflake = snowflake.hm_production_terraform_read_write_role }
  source                  = "../../../../modules/snowflake/hm_snowflake_grant_database_privileges_to_role"
  for_each                = { for x in local.production_department_db_department_name_schema_name_list : "${x.department_name}.${x.schema_name}" => x }
  snowflake_role_name     = "HM_PRODUCTION_DEPARTMENT_${each.value.department_name}_DB_${each.value.schema_name}_READ_ONLY_ROLE"
  privileges              = ["USAGE"]
  snowflake_database_name = "PRODUCTION_DEPARTMENT_${each.value.department_name}_DB"
  depends_on = [
    module.snowflake_production_department_db_schema_read_only_role
  ]
}
module "snowflake_grant_schema_privileges_to_production_department_db_schema_read_only_role" {
  providers               = { snowflake = snowflake.hm_production_terraform_read_write_role }
  source                  = "../../../../modules/snowflake/hm_snowflake_grant_schema_privileges_to_role"
  for_each                = { for x in local.production_department_db_department_name_schema_name_list : "${x.department_name}.${x.schema_name}" => x }
  snowflake_role_name     = "HM_PRODUCTION_DEPARTMENT_${each.value.department_name}_DB_${each.value.schema_name}_READ_ONLY_ROLE"
  privileges              = ["USAGE"]
  snowflake_database_name = "PRODUCTION_DEPARTMENT_${each.value.department_name}_DB"
  snowflake_schema_name   = each.value.schema_name
  depends_on = [
    module.snowflake_production_department_db_schema_read_only_role
  ]
}
module "snowflake_grant_all_future_table_in_schema_privileges_to_production_department_db_schema_read_only_role" {
  providers               = { snowflake = snowflake.hm_production_terraform_read_write_role }
  source                  = "../../../../modules/snowflake/hm_snowflake_grant_future_table_in_schema_privileges_to_role"
  for_each                = { for x in local.production_department_db_department_name_schema_name_list : "${x.department_name}.${x.schema_name}" => x }
  snowflake_role_name     = "HM_PRODUCTION_DEPARTMENT_${each.value.department_name}_DB_${each.value.schema_name}_READ_ONLY_ROLE"
  privileges              = ["SELECT"]
  snowflake_database_name = "PRODUCTION_DEPARTMENT_${each.value.department_name}_DB"
  snowflake_schema_name   = each.value.schema_name
  depends_on = [
    module.snowflake_production_department_db_schema_read_only_role
  ]
}
module "snowflake_grant_warehouse_privileges_to_production_department_db_schema_read_only_role" {
  providers                = { snowflake = snowflake.hm_production_terraform_read_write_role }
  source                   = "../../../../modules/snowflake/hm_snowflake_grant_warehouse_privileges_to_role"
  for_each                 = { for x in local.production_department_db_department_name_schema_name_list : "${x.department_name}.${x.schema_name}" => x }
  snowflake_role_name      = "HM_PRODUCTION_DEPARTMENT_${each.value.department_name}_DB_${each.value.schema_name}_READ_ONLY_ROLE"
  privileges               = ["USAGE"]
  snowflake_warehouse_name = module.snowflake_production_hm_department_wh_warehouse.name
  depends_on = [
    module.snowflake_production_department_db_schema_read_only_role,
    module.snowflake_production_hm_department_wh_warehouse
  ]
}
module "snowflake_grant_production_department_db_schema_read_only_role_to_production_department_db_schema_read_only_user" {
  providers           = { snowflake = snowflake.hm_production_terraform_read_write_role }
  source              = "../../../../modules/snowflake/hm_snowflake_grant_role_to_user"
  for_each            = { for x in local.production_department_db_department_name_schema_name_read_only_user_name_list : "${x.department_name}.${x.schema_name}.${x.read_only_user_name}" => x }
  snowflake_role_name = "HM_PRODUCTION_DEPARTMENT_${each.value.department_name}_DB_${each.value.schema_name}_READ_ONLY_ROLE"
  snowflake_user_name = each.value.read_only_user_name
  depends_on = [
    module.snowflake_grant_warehouse_privileges_to_production_department_db_schema_read_only_role
  ]
}
# Department role - read write role
module "snowflake_production_department_db_schema_read_write_role" {
  providers           = { snowflake = snowflake.hm_production_terraform_read_write_role }
  source              = "../../../../modules/snowflake/hm_snowflake_role"
  for_each            = { for x in local.production_department_db_department_name_schema_name_list : "${x.department_name}.${x.schema_name}" => x }
  snowflake_role_name = "HM_PRODUCTION_DEPARTMENT_${each.value.department_name}_DB_${each.value.schema_name}_READ_WRITE_ROLE"
}
module "snowflake_grant_schema_privileges_to_production_department_db_schema_read_write_role" {
  providers               = { snowflake = snowflake.hm_production_terraform_read_write_role }
  source                  = "../../../../modules/snowflake/hm_snowflake_grant_schema_privileges_to_role"
  for_each                = { for x in local.production_department_db_department_name_schema_name_list : "${x.department_name}.${x.schema_name}" => x }
  snowflake_role_name     = "HM_PRODUCTION_DEPARTMENT_${each.value.department_name}_DB_${each.value.schema_name}_READ_WRITE_ROLE"
  privileges              = ["CREATE TABLE", "CREATE VIEW"]
  snowflake_database_name = "PRODUCTION_DEPARTMENT_${each.value.department_name}_DB"
  snowflake_schema_name   = each.value.schema_name
  depends_on = [
    module.snowflake_production_department_db_schema_read_write_role
  ]
}
module "snowflake_grant_all_future_table_in_schema_privileges_to_production_department_db_schema_read_write_role" {
  providers               = { snowflake = snowflake.hm_production_terraform_read_write_role }
  source                  = "../../../../modules/snowflake/hm_snowflake_grant_future_table_in_schema_privileges_to_role"
  for_each                = { for x in local.production_department_db_department_name_schema_name_list : "${x.department_name}.${x.schema_name}" => x }
  snowflake_role_name     = "HM_PRODUCTION_DEPARTMENT_${each.value.department_name}_DB_${each.value.schema_name}_READ_WRITE_ROLE"
  privileges              = ["INSERT", "UPDATE", "DELETE"]
  snowflake_database_name = "PRODUCTION_DEPARTMENT_${each.value.department_name}_DB"
  snowflake_schema_name   = each.value.schema_name
  depends_on = [
    module.snowflake_production_department_db_schema_read_write_role
  ]
}
module "snowflake_grant_production_department_db_schema_read_only_role_to_production_department_db_schema_read_write_role" {
  providers                    = { snowflake = snowflake.hm_production_terraform_read_write_role }
  source                       = "../../../../modules/snowflake/hm_snowflake_grant_role_to_role"
  for_each                     = { for x in local.production_department_db_department_name_schema_name_list : "${x.department_name}.${x.schema_name}" => x }
  snowflake_role_name          = "HM_PRODUCTION_DEPARTMENT_${each.value.department_name}_DB_${each.value.schema_name}_READ_ONLY_ROLE"
  snowflake_grant_to_role_name = "HM_PRODUCTION_DEPARTMENT_${each.value.department_name}_DB_${each.value.schema_name}_READ_WRITE_ROLE"
  depends_on = [
    module.snowflake_production_department_db_schema_read_only_role,
    module.snowflake_grant_database_privileges_to_production_department_db_schema_read_only_role,
    module.snowflake_grant_all_future_table_in_schema_privileges_to_production_department_db_schema_read_only_role,
    module.snowflake_grant_warehouse_privileges_to_production_department_db_schema_read_only_role,
    module.snowflake_production_department_db_schema_read_write_role
  ]
}
module "snowflake_grant_production_department_db_schema_read_write_role_to_production_department_db_schema_read_write_user" {
  providers           = { snowflake = snowflake.hm_production_terraform_read_write_role }
  source              = "../../../../modules/snowflake/hm_snowflake_grant_role_to_user"
  for_each            = { for x in local.production_department_db_department_name_schema_name_read_write_user_name_list : "${x.department_name}.${x.schema_name}.${x.read_write_user_name}" => x }
  snowflake_role_name = "HM_PRODUCTION_DEPARTMENT_${each.value.department_name}_DB_${each.value.schema_name}_READ_WRITE_ROLE"
  snowflake_user_name = each.value.read_write_user_name
  depends_on = [
    module.snowflake_production_department_db_schema_read_only_role
  ]
}
# Department role - admin role
module "snowflake_production_department_db_admin_role" {
  providers           = { snowflake = snowflake.hm_production_terraform_read_write_role }
  source              = "../../../../modules/snowflake/hm_snowflake_role"
  for_each            = local.production_department_db_department_names
  snowflake_role_name = "HM_PRODUCTION_DEPARTMENT_${each.value}_DB_ADMIN_ROLE"
}
module "snowflake_grant_database_all_privileges_to_role_to_production_department_db_admin_role" {
  providers               = { snowflake = snowflake.hm_production_terraform_read_write_role }
  source                  = "../../../../modules/snowflake/hm_snowflake_grant_database_all_privileges_to_role"
  for_each                = local.production_department_db_department_names
  snowflake_role_name     = "HM_PRODUCTION_DEPARTMENT_${each.value}_DB_ADMIN_ROLE"
  snowflake_database_name = "PRODUCTION_DEPARTMENT_${each.value}_DB"
  depends_on = [
    module.snowflake_production_department_db_admin_role
  ]
}
module "snowflake_grant_existing_schema_all_privileges_to_role_to_production_department_db_admin_role" {
  providers               = { snowflake = snowflake.hm_production_terraform_read_write_role }
  source                  = "../../../../modules/snowflake/hm_snowflake_grant_existing_schema_all_privileges_to_role"
  for_each                = local.production_department_db_department_names
  snowflake_role_name     = "HM_PRODUCTION_DEPARTMENT_${each.value}_DB_ADMIN_ROLE"
  snowflake_database_name = "PRODUCTION_DEPARTMENT_${each.value}_DB"
  depends_on = [
    module.snowflake_production_department_db_admin_role
  ]
}
module "snowflake_grant_future_schema_all_privileges_to_role_to_production_department_db_admin_role" {
  providers               = { snowflake = snowflake.hm_production_terraform_read_write_role }
  source                  = "../../../../modules/snowflake/hm_snowflake_grant_future_schema_all_privileges_to_role"
  for_each                = local.production_department_db_department_names
  snowflake_role_name     = "HM_PRODUCTION_DEPARTMENT_${each.value}_DB_ADMIN_ROLE"
  snowflake_database_name = "PRODUCTION_DEPARTMENT_${each.value}_DB"
  depends_on = [
    module.snowflake_production_department_db_admin_role
  ]
}
module "snowflake_grant_existing_table_all_privileges_to_role_to_production_department_db_admin_role" {
  providers               = { snowflake = snowflake.hm_production_terraform_read_write_role }
  source                  = "../../../../modules/snowflake/hm_snowflake_grant_existing_table_all_privileges_to_role"
  for_each                = local.production_department_db_department_names
  snowflake_role_name     = "HM_PRODUCTION_DEPARTMENT_${each.value}_DB_ADMIN_ROLE"
  snowflake_database_name = "PRODUCTION_DEPARTMENT_${each.value}_DB"
  depends_on = [
    module.snowflake_production_department_db_admin_role
  ]
}
module "snowflake_grant_future_table_all_privileges_to_role_to_production_department_db_admin_role" {
  providers               = { snowflake = snowflake.hm_production_terraform_read_write_role }
  source                  = "../../../../modules/snowflake/hm_snowflake_grant_future_table_all_privileges_to_role"
  for_each                = local.production_department_db_department_names
  snowflake_role_name     = "HM_PRODUCTION_DEPARTMENT_${each.value}_DB_ADMIN_ROLE"
  snowflake_database_name = "PRODUCTION_DEPARTMENT_${each.value}_DB"
  depends_on = [
    module.snowflake_production_department_db_admin_role
  ]
}
module "snowflake_grant_production_department_db_schema_read_write_role_to_production_department_db_admin_role" {
  providers                    = { snowflake = snowflake.hm_production_terraform_read_write_role }
  source                       = "../../../../modules/snowflake/hm_snowflake_grant_role_to_role"
  for_each                     = { for x in local.production_department_db_department_name_schema_name_list : "${x.department_name}.${x.schema_name}" => x }
  snowflake_role_name          = "HM_PRODUCTION_DEPARTMENT_${each.value.department_name}_DB_${each.value.schema_name}_READ_WRITE_ROLE"
  snowflake_grant_to_role_name = "HM_PRODUCTION_DEPARTMENT_${each.value.department_name}_DB_ADMIN_ROLE"
  depends_on = [
    module.snowflake_production_department_db_schema_read_write_role,
    module.snowflake_grant_production_department_db_schema_read_only_role_to_production_department_db_schema_read_write_role,
    module.snowflake_grant_schema_privileges_to_production_department_db_schema_read_write_role,
    module.snowflake_grant_all_future_table_in_schema_privileges_to_production_department_db_schema_read_write_role,
    module.snowflake_production_department_db_admin_role
  ]
}
module "snowflake_grant_production_department_db_schema_admin_role_to_production_department_db_schema_admin_user" {
  providers           = { snowflake = snowflake.hm_production_terraform_read_write_role }
  source              = "../../../../modules/snowflake/hm_snowflake_grant_role_to_user"
  for_each            = { for x in local.production_department_db_department_name_admin_user_name_list : "${x.department_name}.${x.admin_user_name}" => x }
  snowflake_role_name = "HM_PRODUCTION_DEPARTMENT_${each.value.department_name}_DB_ADMIN_ROLE"
  snowflake_user_name = each.value.admin_user_name
  depends_on = [
    module.snowflake_production_department_db_admin_role
  ]
}

# HM Airbyte warehouse
module "snowflake_production_hm_airbyte_wh_warehouse" {
  providers                = { snowflake = snowflake.hm_production_terraform_read_write_role }
  source                   = "../../../../modules/snowflake/hm_snowflake_warehouse"
  snowflake_warehouse_name = "HM_PRODUCTION_HM_AIRBYTE_WH"
  snowflake_warehouse_size = "xsmall"
  auto_suspend_min         = var.production_airbyte_warehouse_auto_suspend_min
}
# HM Airbyte role - owner role
# https://docs.airbyte.com/integrations/destinations/snowflake
module "snowflake_production_hm_airbyte_db_owner_role" {
  providers           = { snowflake = snowflake.hm_production_terraform_read_write_role }
  source              = "../../../../modules/snowflake/hm_snowflake_role"
  snowflake_role_name = "HM_PRODUCTION_HM_AIRBYTE_DB_OWNER_ROLE"
}
module "snowflake_transfer_production_hm_airbyte_db_ownership_to_production_hm_airbyte_db_owner_role" {
  providers               = { snowflake = snowflake.hm_production_terraform_read_write_role }
  source                  = "../../../../modules/snowflake/hm_snowflake_transfer_database_ownership_to_role"
  snowflake_database_name = "PRODUCTION_HM_AIRBYTE_DB"
  snowflake_role_name     = module.snowflake_production_hm_airbyte_db_owner_role.name
  depends_on = [
    module.snowflake_production_hm_airbyte_db_owner_role
  ]
}
# HM Airbyte role - owner user
module "snowflake_production_hm_airbyte_db_owner_user" {
  providers                                 = { snowflake = snowflake.hm_production_terraform_read_write_role }
  source                                    = "../../../../modules/snowflake/hm_snowflake_user"
  snowflake_user_name                       = "HM_PRODUCTION_HM_AIRBYTE_DB_OWNER_USER"
  default_role                              = module.snowflake_production_hm_airbyte_db_owner_role.name
  default_warehouse                         = module.snowflake_production_hm_airbyte_wh_warehouse.name
  rsa_public_key_without_header_and_trailer = var.production_hm_airbyte_db_owner_user_rsa_public_key_without_header_and_trailer
  depends_on = [
    module.snowflake_production_hm_airbyte_db_owner_role
  ]
}
module "snowflake_grant_production_hm_airbyte_db_owner_role_to_production_hm_airbyte_db_owner_user" {
  providers           = { snowflake = snowflake.hm_production_terraform_read_write_role }
  source              = "../../../../modules/snowflake/hm_snowflake_grant_role_to_user"
  snowflake_role_name = module.snowflake_production_hm_airbyte_db_owner_role.name
  snowflake_user_name = module.snowflake_production_hm_airbyte_db_owner_user.name
  depends_on = [
    module.snowflake_production_hm_airbyte_db_owner_user,
    module.snowflake_production_hm_airbyte_db_owner_role
  ]
}

# HM Kafka warehouse
module "snowflake_production_hm_kafka_wh_warehouse" {
  providers                = { snowflake = snowflake.hm_production_terraform_read_write_role }
  source                   = "../../../../modules/snowflake/hm_snowflake_warehouse"
  snowflake_warehouse_name = "HM_PRODUCTION_HM_KAFKA_WH"
  snowflake_warehouse_size = "xsmall"
  auto_suspend_min         = var.production_kafka_warehouse_auto_suspend_min
}
# HM Kafka role
locals {
  production_hm_kafka_db_department_names = toset([for department in var.production_hm_kafka_db_departments : department.name])
  production_hm_kafka_db_department_name_read_write_user_rsa_public_key_without_header_and_trailer_list = flatten([
    for department in var.production_hm_kafka_db_departments :
    {
      department_name                                           = department.name
      read_write_user_rsa_public_key_without_header_and_trailer = department.read_write_user_rsa_public_key_without_header_and_trailer
    }
  ])
}
# HM Kafka role - read only role
module "snowflake_production_hm_kafka_db_department_read_only_role" {
  providers           = { snowflake = snowflake.hm_production_terraform_read_write_role }
  source              = "../../../../modules/snowflake/hm_snowflake_role"
  for_each            = local.production_hm_kafka_db_department_names
  snowflake_role_name = "HM_PRODUCTION_HM_KAFKA_DB_${each.value}_READ_ONLY_ROLE"
}
module "snowflake_grant_database_privileges_to_production_hm_kafka_db_department_read_only_role" {
  providers               = { snowflake = snowflake.hm_production_terraform_read_write_role }
  source                  = "../../../../modules/snowflake/hm_snowflake_grant_database_privileges_to_role"
  for_each                = local.production_hm_kafka_db_department_names
  snowflake_role_name     = "HM_PRODUCTION_HM_KAFKA_DB_${each.value}_READ_ONLY_ROLE"
  privileges              = ["USAGE"]
  snowflake_database_name = "PRODUCTION_HM_KAFKA_DB"
  depends_on = [
    module.snowflake_production_hm_kafka_db_department_read_only_role
  ]
}
module "snowflake_grant_schema_privileges_to_production_hm_kafka_db_department_read_only_role" {
  providers               = { snowflake = snowflake.hm_production_terraform_read_write_role }
  source                  = "../../../../modules/snowflake/hm_snowflake_grant_schema_privileges_to_role"
  for_each                = local.production_hm_kafka_db_department_names
  snowflake_role_name     = "HM_PRODUCTION_HM_KAFKA_DB_${each.value}_READ_ONLY_ROLE"
  privileges              = ["USAGE"]
  snowflake_database_name = "PRODUCTION_HM_KAFKA_DB"
  snowflake_schema_name   = each.value
  depends_on = [
    module.snowflake_production_hm_kafka_db_department_read_only_role
  ]
}
module "snowflake_grant_warehouse_privileges_to_production_hm_kafka_db_department_read_only_role" {
  providers                = { snowflake = snowflake.hm_production_terraform_read_write_role }
  source                   = "../../../../modules/snowflake/hm_snowflake_grant_warehouse_privileges_to_role"
  for_each                 = local.production_hm_kafka_db_department_names
  snowflake_role_name      = "HM_PRODUCTION_HM_KAFKA_DB_${each.value}_READ_ONLY_ROLE"
  privileges               = ["USAGE"]
  snowflake_warehouse_name = module.snowflake_production_hm_kafka_wh_warehouse.name
  depends_on = [
    module.snowflake_production_hm_kafka_db_department_read_only_role,
    module.snowflake_production_hm_kafka_wh_warehouse
  ]
}
# HM Kafka role - read write role
module "snowflake_production_hm_kafka_db_department_read_write_role" {
  providers           = { snowflake = snowflake.hm_production_terraform_read_write_role }
  source              = "../../../../modules/snowflake/hm_snowflake_role"
  for_each            = local.production_hm_kafka_db_department_names
  snowflake_role_name = "HM_PRODUCTION_HM_KAFKA_DB_${each.value}_READ_WRITE_ROLE"
}
# https://docs.snowflake.com/en/user-guide/kafka-connector-install
module "snowflake_grant_schema_privileges_to_production_hm_kafka_db_department_read_write_role" {
  providers               = { snowflake = snowflake.hm_production_terraform_read_write_role }
  source                  = "../../../../modules/snowflake/hm_snowflake_grant_schema_privileges_to_role"
  for_each                = local.production_hm_kafka_db_department_names
  snowflake_role_name     = "HM_PRODUCTION_HM_KAFKA_DB_${each.value}_READ_WRITE_ROLE"
  privileges              = ["CREATE TABLE", "CREATE STAGE", "CREATE PIPE"]
  snowflake_database_name = "PRODUCTION_HM_KAFKA_DB"
  snowflake_schema_name   = each.value
  depends_on = [
    module.snowflake_production_hm_kafka_db_department_read_write_role
  ]
}
module "snowflake_grant_production_hm_kafka_db_department_read_only_role_to_production_hm_kafka_db_department_read_write_role" {
  providers                    = { snowflake = snowflake.hm_production_terraform_read_write_role }
  source                       = "../../../../modules/snowflake/hm_snowflake_grant_role_to_role"
  for_each                     = local.production_hm_kafka_db_department_names
  snowflake_role_name          = "HM_PRODUCTION_HM_KAFKA_DB_${each.value}_READ_ONLY_ROLE"
  snowflake_grant_to_role_name = "HM_PRODUCTION_HM_KAFKA_DB_${each.value}_READ_WRITE_ROLE"
  depends_on = [
    module.snowflake_production_hm_kafka_db_department_read_only_role,
    module.snowflake_production_hm_kafka_db_department_read_write_role
  ]
}
# HM Kafka role - read write user
module "snowflake_production_hm_kafka_db_department_read_write_user" {
  providers                                 = { snowflake = snowflake.hm_production_terraform_read_write_role }
  source                                    = "../../../../modules/snowflake/hm_snowflake_user"
  for_each                                  = { for x in local.production_hm_kafka_db_department_name_read_write_user_rsa_public_key_without_header_and_trailer_list : "${x.department_name}.${x.read_write_user_rsa_public_key_without_header_and_trailer}" => x }
  snowflake_user_name                       = "HM_PRODUCTION_HM_KAFKA_DB_${each.value.department_name}_READ_WRITE_USER"
  default_role                              = "HM_PRODUCTION_HM_KAFKA_DB_${each.value.department_name}_READ_WRITE_ROLE"
  default_warehouse                         = module.snowflake_production_hm_kafka_wh_warehouse.name
  rsa_public_key_without_header_and_trailer = each.value.read_write_user_rsa_public_key_without_header_and_trailer
  depends_on = [
    module.snowflake_production_hm_kafka_db_department_read_write_role
  ]
}
module "snowflake_grant_production_hm_kafka_db_department_read_write_role_to_production_hm_kafka_db_department_read_write_user" {
  providers           = { snowflake = snowflake.hm_production_terraform_read_write_role }
  source              = "../../../../modules/snowflake/hm_snowflake_grant_role_to_user"
  for_each            = local.production_hm_kafka_db_department_names
  snowflake_role_name = "HM_PRODUCTION_HM_KAFKA_DB_${each.value}_READ_WRITE_ROLE"
  snowflake_user_name = "HM_PRODUCTION_HM_KAFKA_DB_${each.value}_READ_WRITE_USER"
  depends_on = [
    module.snowflake_production_hm_kafka_db_department_read_write_user,
    module.snowflake_production_hm_kafka_db_department_read_write_role
  ]
}

# HM Streamlit warehouse
module "snowflake_production_hm_streamlit_wh_warehouse" {
  providers                = { snowflake = snowflake.hm_production_terraform_read_write_role }
  source                   = "../../../../modules/snowflake/hm_snowflake_warehouse"
  snowflake_warehouse_name = "HM_PRODUCTION_HM_STREAMLIT_WH"
  snowflake_warehouse_size = "xsmall"
  auto_suspend_min         = var.production_streamlit_warehouse_auto_suspend_min
}
# HM Streamlit role - read write role
module "snowflake_production_hm_streamlit_db_department_read_write_role" {
  providers           = { snowflake = snowflake.hm_production_terraform_read_write_role }
  source              = "../../../../modules/snowflake/hm_snowflake_role"
  for_each            = local.production_department_db_department_names
  snowflake_role_name = "HM_PRODUCTION_HM_STREAMLIT_DB_${each.value}_READ_WRITE_ROLE"
}
module "snowflake_grant_database_privileges_to_production_hm_streamlit_db_read_write_role" {
  providers               = { snowflake = snowflake.hm_production_terraform_read_write_role }
  source                  = "../../../../modules/snowflake/hm_snowflake_grant_database_privileges_to_role"
  for_each                = { for x in local.production_department_db_department_name_schema_name_list : "${x.department_name}.${x.schema_name}" => x }
  snowflake_role_name     = "HM_PRODUCTION_HM_STREAMLIT_DB_${each.value.department_name}_READ_WRITE_ROLE"
  privileges              = ["USAGE"]
  snowflake_database_name = "PRODUCTION_HM_STREAMLIT_DB"
  depends_on = [
    module.snowflake_production_hm_streamlit_db_department_read_write_role
  ]
}
module "snowflake_grant_schema_privileges_to_production_hm_streamlit_db_read_write_role" {
  providers               = { snowflake = snowflake.hm_production_terraform_read_write_role }
  source                  = "../../../../modules/snowflake/hm_snowflake_grant_schema_privileges_to_role"
  for_each                = { for x in local.production_department_db_department_name_schema_name_list : "${x.department_name}.${x.schema_name}" => x }
  snowflake_role_name     = "HM_PRODUCTION_HM_STREAMLIT_DB_${each.value.department_name}_READ_WRITE_ROLE"
  privileges              = ["USAGE", "CREATE STREAMLIT", "CREATE STAGE"]
  snowflake_database_name = "PRODUCTION_HM_STREAMLIT_DB"
  snowflake_schema_name   = each.value.department_name
  depends_on = [
    module.snowflake_production_hm_streamlit_db_department_read_write_role
  ]
}
module "snowflake_grant_warehouse_privileges_to_production_department_db_department_read_write_role" {
  providers                = { snowflake = snowflake.hm_production_terraform_read_write_role }
  source                   = "../../../../modules/snowflake/hm_snowflake_grant_warehouse_privileges_to_role"
  for_each                 = { for x in local.production_department_db_department_name_schema_name_list : "${x.department_name}.${x.schema_name}" => x }
  snowflake_role_name      = "HM_PRODUCTION_HM_STREAMLIT_DB_${each.value.department_name}_READ_WRITE_ROLE"
  privileges               = ["USAGE"]
  snowflake_warehouse_name = module.snowflake_production_hm_streamlit_wh_warehouse.name
  depends_on = [
    module.snowflake_production_hm_streamlit_db_department_read_write_role,
    module.snowflake_production_hm_streamlit_wh_warehouse
  ]
}
# HM Streamlit role - end user role
# Empty department role to help share Streamlit app to different department end users
module "snowflake_production_streamlit_app_for_department_end_users_role" {
  providers           = { snowflake = snowflake.hm_production_terraform_read_write_role }
  source              = "../../../../modules/snowflake/hm_snowflake_role"
  for_each            = local.production_department_db_department_names
  snowflake_role_name = "HM_PRODUCTION_STREAMLIT_APP_FOR_DEPARTMENT_${each.value}_END_USERS_ROLE"
}
