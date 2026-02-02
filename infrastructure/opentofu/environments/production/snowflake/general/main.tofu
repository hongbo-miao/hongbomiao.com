data "terraform_remote_state" "hm_terraform_remote_state_snowflake_data" {
  backend = "s3"
  config = {
    region = "us-west-2"
    bucket = "hm-terraform-hongbomiao"
    key    = "production/snowflake/storage/terraform.tfstate"
  }
}

# Department warehouse
module "department_wh_warehouse" {
  providers                = { snowflake = snowflake.opentofu_role }
  source                   = "../../../../modules/snowflake/snowflake_warehouse"
  snowflake_warehouse_name = "${var.environment}_DEPARTMENT_WH"
  snowflake_warehouse_size = "xsmall"
  auto_suspend_min         = var.department_warehouse_auto_suspend_min
}
# Department role
locals {
  department_db_department_names = toset([for department in var.department_db_departments : department.name])
  department_db_department_name_schema_name_list = flatten([
    for department in var.department_db_departments :
    [
      for schema in department.schemas :
      {
        department_name                           = department.name
        admin_user_names                          = department.admin_user_names
        schema_name                               = schema.name
        read_only_service_account_rsa_public_key  = schema.read_only_service_account_rsa_public_key
        read_write_service_account_rsa_public_key = schema.read_write_service_account_rsa_public_key
      }
    ]
  ])
  department_db_department_name_admin_user_name_list = flatten([
    for department in var.department_db_departments :
    [
      for admin_user_name in department.admin_user_names :
      {
        department_name = department.name
        admin_user_name = admin_user_name
      }
    ]
  ])
  department_db_department_name_schema_name_read_only_user_name_list = flatten([
    for department in var.department_db_departments :
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
  department_db_department_name_schema_name_read_write_user_name_list = flatten([
    for department in var.department_db_departments :
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
module "department_db_schema_read_only_role" {
  providers           = { snowflake = snowflake.opentofu_role }
  source              = "../../../../modules/snowflake/snowflake_account_role"
  for_each            = { for x in local.department_db_department_name_schema_name_list : "${x.department_name}.${x.schema_name}" => x }
  snowflake_role_name = "${var.environment}_${each.value.department_name}_DB_${each.value.schema_name}_READ_ONLY_ROLE"
}
module "grant_database_privileges_to_department_db_schema_read_only_role" {
  providers               = { snowflake = snowflake.opentofu_role }
  source                  = "../../../../modules/snowflake/snowflake_grant_database_privileges_to_role"
  for_each                = { for x in local.department_db_department_name_schema_name_list : "${x.department_name}.${x.schema_name}" => x }
  snowflake_role_name     = "${var.environment}_${each.value.department_name}_DB_${each.value.schema_name}_READ_ONLY_ROLE"
  privileges              = ["USAGE"]
  snowflake_database_name = "${var.environment}_${each.value.department_name}_DB"
  depends_on = [
    module.department_db_schema_read_only_role
  ]
}
module "grant_schema_privileges_to_department_db_schema_read_only_role" {
  providers               = { snowflake = snowflake.opentofu_role }
  source                  = "../../../../modules/snowflake/snowflake_grant_schema_privileges_to_role"
  for_each                = { for x in local.department_db_department_name_schema_name_list : "${x.department_name}.${x.schema_name}" => x }
  snowflake_role_name     = "${var.environment}_${each.value.department_name}_DB_${each.value.schema_name}_READ_ONLY_ROLE"
  privileges              = ["USAGE"]
  snowflake_database_name = "${var.environment}_${each.value.department_name}_DB"
  snowflake_schema_name   = each.value.schema_name
  depends_on = [
    module.department_db_schema_read_only_role
  ]
}
module "grant_table_in_schema_privileges_to_department_db_schema_read_only_role" {
  providers               = { snowflake = snowflake.opentofu_role }
  source                  = "../../../../modules/snowflake/snowflake_grant_table_in_schema_privileges_to_role"
  for_each                = { for x in local.department_db_department_name_schema_name_list : "${x.department_name}.${x.schema_name}" => x }
  snowflake_role_name     = "${var.environment}_${each.value.department_name}_DB_${each.value.schema_name}_READ_ONLY_ROLE"
  privileges              = ["SELECT"]
  snowflake_database_name = "${var.environment}_${each.value.department_name}_DB"
  snowflake_schema_name   = each.value.schema_name
  depends_on = [
    module.department_db_schema_read_only_role
  ]
}
module "grant_view_in_schema_privileges_to_department_db_schema_read_only_role" {
  providers               = { snowflake = snowflake.opentofu_role }
  source                  = "../../../../modules/snowflake/snowflake_grant_view_in_schema_privileges_to_role"
  for_each                = { for x in local.department_db_department_name_schema_name_list : "${x.department_name}.${x.schema_name}" => x }
  snowflake_role_name     = "${var.environment}_${each.value.department_name}_DB_${each.value.schema_name}_READ_ONLY_ROLE"
  privileges              = ["SELECT"]
  snowflake_database_name = "${var.environment}_${each.value.department_name}_DB"
  snowflake_schema_name   = each.value.schema_name
  depends_on = [
    module.department_db_schema_read_only_role
  ]
}
module "grant_warehouse_privileges_to_department_db_schema_read_only_role" {
  providers                = { snowflake = snowflake.opentofu_role }
  source                   = "../../../../modules/snowflake/snowflake_grant_warehouse_privileges_to_role"
  for_each                 = { for x in local.department_db_department_name_schema_name_list : "${x.department_name}.${x.schema_name}" => x }
  snowflake_role_name      = "${var.environment}_${each.value.department_name}_DB_${each.value.schema_name}_READ_ONLY_ROLE"
  privileges               = ["USAGE"]
  snowflake_warehouse_name = module.department_wh_warehouse.name
  depends_on = [
    module.department_db_schema_read_only_role,
    module.department_wh_warehouse
  ]
}
module "grant_department_db_schema_read_only_role_to_department_db_schema_read_only_user" {
  providers           = { snowflake = snowflake.opentofu_role }
  source              = "../../../../modules/snowflake/snowflake_grant_role_to_user"
  for_each            = { for x in local.department_db_department_name_schema_name_read_only_user_name_list : "${x.department_name}.${x.schema_name}.${x.read_only_user_name}" => x }
  snowflake_role_name = "${var.environment}_${each.value.department_name}_DB_${each.value.schema_name}_READ_ONLY_ROLE"
  snowflake_user_name = each.value.read_only_user_name
  depends_on = [
    module.department_db_schema_read_only_role
  ]
}
# Department role - read only service account
module "department_db_schema_read_only_service_account" {
  providers           = { snowflake = snowflake.opentofu_role }
  source              = "../../../../modules/snowflake/snowflake_user"
  for_each            = { for x in local.department_db_department_name_schema_name_list : "${x.department_name}.${x.read_only_service_account_rsa_public_key}" => x if x.read_only_service_account_rsa_public_key != null }
  snowflake_user_name = "${var.environment}_${each.value.department_name}_DB_${each.value.schema_name}_READ_ONLY_SERVICE_ACCOUNT"
  default_role        = "${var.environment}_${each.value.department_name}_DB_${each.value.schema_name}_READ_ONLY_ROLE"
  default_warehouse   = module.hm_kafka_wh_warehouse.name
  rsa_public_key      = each.value.read_only_service_account_rsa_public_key
  depends_on = [
    module.department_db_schema_read_only_role
  ]
}
# Department role - read write role
module "department_db_schema_read_write_role" {
  providers           = { snowflake = snowflake.opentofu_role }
  source              = "../../../../modules/snowflake/snowflake_account_role"
  for_each            = { for x in local.department_db_department_name_schema_name_list : "${x.department_name}.${x.schema_name}" => x }
  snowflake_role_name = "${var.environment}_${each.value.department_name}_DB_${each.value.schema_name}_READ_WRITE_ROLE"
}
module "grant_schema_privileges_to_department_db_schema_read_write_role" {
  providers               = { snowflake = snowflake.opentofu_role }
  source                  = "../../../../modules/snowflake/snowflake_grant_schema_privileges_to_role"
  for_each                = { for x in local.department_db_department_name_schema_name_list : "${x.department_name}.${x.schema_name}" => x }
  snowflake_role_name     = "${var.environment}_${each.value.department_name}_DB_${each.value.schema_name}_READ_WRITE_ROLE"
  privileges              = ["CREATE TABLE", "CREATE VIEW"]
  snowflake_database_name = "${var.environment}_${each.value.department_name}_DB"
  snowflake_schema_name   = each.value.schema_name
  depends_on = [
    module.department_db_schema_read_write_role
  ]
}
module "grant_table_in_schema_privileges_to_department_db_schema_read_write_role" {
  providers               = { snowflake = snowflake.opentofu_role }
  source                  = "../../../../modules/snowflake/snowflake_grant_table_in_schema_privileges_to_role"
  for_each                = { for x in local.department_db_department_name_schema_name_list : "${x.department_name}.${x.schema_name}" => x }
  snowflake_role_name     = "${var.environment}_${each.value.department_name}_DB_${each.value.schema_name}_READ_WRITE_ROLE"
  privileges              = ["INSERT", "UPDATE", "DELETE"]
  snowflake_database_name = "${var.environment}_${each.value.department_name}_DB"
  snowflake_schema_name   = each.value.schema_name
  depends_on = [
    module.department_db_schema_read_write_role
  ]
}
module "grant_view_in_schema_privileges_to_department_db_schema_read_write_role" {
  providers               = { snowflake = snowflake.opentofu_role }
  source                  = "../../../../modules/snowflake/snowflake_grant_view_in_schema_privileges_to_role"
  for_each                = { for x in local.department_db_department_name_schema_name_list : "${x.department_name}.${x.schema_name}" => x }
  snowflake_role_name     = "${var.environment}_${each.value.department_name}_DB_${each.value.schema_name}_READ_WRITE_ROLE"
  privileges              = ["INSERT", "UPDATE", "DELETE"]
  snowflake_database_name = "${var.environment}_${each.value.department_name}_DB"
  snowflake_schema_name   = each.value.schema_name
  depends_on = [
    module.department_db_schema_read_write_role
  ]
}
module "grant_department_db_schema_read_only_role_to_department_db_schema_read_write_role" {
  providers                    = { snowflake = snowflake.opentofu_role }
  source                       = "../../../../modules/snowflake/snowflake_grant_role_to_role"
  for_each                     = { for x in local.department_db_department_name_schema_name_list : "${x.department_name}.${x.schema_name}" => x }
  snowflake_role_name          = "${var.environment}_${each.value.department_name}_DB_${each.value.schema_name}_READ_ONLY_ROLE"
  snowflake_grant_to_role_name = "${var.environment}_${each.value.department_name}_DB_${each.value.schema_name}_READ_WRITE_ROLE"
  depends_on = [
    module.department_db_schema_read_only_role,
    module.department_db_schema_read_write_role
  ]
}
module "grant_department_db_schema_read_write_role_to_department_db_schema_read_write_user" {
  providers           = { snowflake = snowflake.opentofu_role }
  source              = "../../../../modules/snowflake/snowflake_grant_role_to_user"
  for_each            = { for x in local.department_db_department_name_schema_name_read_write_user_name_list : "${x.department_name}.${x.schema_name}.${x.read_write_user_name}" => x }
  snowflake_role_name = "${var.environment}_${each.value.department_name}_DB_${each.value.schema_name}_READ_WRITE_ROLE"
  snowflake_user_name = each.value.read_write_user_name
  depends_on = [
    module.department_db_schema_read_only_role
  ]
}
# Department role - read write service account
module "department_db_schema_read_write_service_account" {
  providers           = { snowflake = snowflake.opentofu_role }
  source              = "../../../../modules/snowflake/snowflake_user"
  for_each            = { for x in local.department_db_department_name_schema_name_list : "${x.department_name}.${x.read_write_service_account_rsa_public_key}" => x if x.read_write_service_account_rsa_public_key != null }
  snowflake_user_name = "${var.environment}_${each.value.department_name}_DB_${each.value.schema_name}_READ_WRITE_SERVICE_ACCOUNT"
  default_role        = "${var.environment}_${each.value.department_name}_DB_${each.value.schema_name}_READ_WRITE_ROLE"
  default_warehouse   = module.hm_kafka_wh_warehouse.name
  rsa_public_key      = each.value.read_write_service_account_rsa_public_key
  depends_on = [
    module.department_db_schema_read_write_role
  ]
}
# Department role - admin role
module "department_db_admin_role" {
  providers           = { snowflake = snowflake.opentofu_role }
  source              = "../../../../modules/snowflake/snowflake_account_role"
  for_each            = local.department_db_department_names
  snowflake_role_name = "${var.environment}_${each.value}_DB_ADMIN_ROLE"
}
module "grant_database_all_privileges_to_role_to_department_db_admin_role" {
  providers               = { snowflake = snowflake.opentofu_role }
  source                  = "../../../../modules/snowflake/snowflake_grant_database_all_privileges_to_role"
  for_each                = local.department_db_department_names
  snowflake_role_name     = "${var.environment}_${each.value}_DB_ADMIN_ROLE"
  snowflake_database_name = "${var.environment}_${each.value}_DB"
  depends_on = [
    module.department_db_admin_role
  ]
}
module "grant_schema_all_privileges_to_role_to_department_db_admin_role" {
  providers               = { snowflake = snowflake.opentofu_role }
  source                  = "../../../../modules/snowflake/snowflake_grant_schema_all_privileges_to_role"
  for_each                = local.department_db_department_names
  snowflake_role_name     = "${var.environment}_${each.value}_DB_ADMIN_ROLE"
  snowflake_database_name = "${var.environment}_${each.value}_DB"
  depends_on = [
    module.department_db_admin_role
  ]
}
module "grant_table_all_privileges_to_role_to_department_db_admin_role" {
  providers               = { snowflake = snowflake.opentofu_role }
  source                  = "../../../../modules/snowflake/snowflake_grant_table_all_privileges_to_role"
  for_each                = local.department_db_department_names
  snowflake_role_name     = "${var.environment}_${each.value}_DB_ADMIN_ROLE"
  snowflake_database_name = "${var.environment}_${each.value}_DB"
  depends_on = [
    module.department_db_admin_role
  ]
}
module "grant_view_all_privileges_to_role_to_department_db_admin_role" {
  providers               = { snowflake = snowflake.opentofu_role }
  source                  = "../../../../modules/snowflake/snowflake_grant_view_all_privileges_to_role"
  for_each                = local.department_db_department_names
  snowflake_role_name     = "${var.environment}_${each.value}_DB_ADMIN_ROLE"
  snowflake_database_name = "${var.environment}_${each.value}_DB"
  depends_on = [
    module.department_db_admin_role
  ]
}
module "grant_department_db_schema_read_write_role_to_department_db_admin_role" {
  providers                    = { snowflake = snowflake.opentofu_role }
  source                       = "../../../../modules/snowflake/snowflake_grant_role_to_role"
  for_each                     = { for x in local.department_db_department_name_schema_name_list : "${x.department_name}.${x.schema_name}" => x }
  snowflake_role_name          = "${var.environment}_${each.value.department_name}_DB_${each.value.schema_name}_READ_WRITE_ROLE"
  snowflake_grant_to_role_name = "${var.environment}_${each.value.department_name}_DB_ADMIN_ROLE"
  depends_on = [
    module.department_db_schema_read_write_role,
    module.department_db_admin_role
  ]
}
module "grant_department_db_schema_admin_role_to_department_db_schema_admin_user" {
  providers           = { snowflake = snowflake.opentofu_role }
  source              = "../../../../modules/snowflake/snowflake_grant_role_to_user"
  for_each            = { for x in local.department_db_department_name_admin_user_name_list : "${x.department_name}.${x.admin_user_name}" => x }
  snowflake_role_name = "${var.environment}_${each.value.department_name}_DB_ADMIN_ROLE"
  snowflake_user_name = each.value.admin_user_name
  depends_on = [
    module.department_db_admin_role
  ]
}

# HM Airbyte warehouse
module "hm_airbyte_wh_warehouse" {
  providers                = { snowflake = snowflake.opentofu_role }
  source                   = "../../../../modules/snowflake/snowflake_warehouse"
  snowflake_warehouse_name = "${var.environment}_HM_AIRBYTE_WH"
  snowflake_warehouse_size = "xsmall"
  auto_suspend_min         = var.airbyte_warehouse_auto_suspend_min
}
# HM Airbyte role - owner role
# https://docs.airbyte.com/integrations/destinations/snowflake
module "hm_airbyte_db_owner_role" {
  providers           = { snowflake = snowflake.opentofu_role }
  source              = "../../../../modules/snowflake/snowflake_account_role"
  snowflake_role_name = "${var.environment}_HM_AIRBYTE_DB_OWNER_ROLE"
}
module "snowflake_transfer_hm_airbyte_db_ownership_to_hm_airbyte_db_owner_role" {
  providers               = { snowflake = snowflake.opentofu_role }
  source                  = "../../../../modules/snowflake/snowflake_transfer_database_ownership_to_role"
  snowflake_database_name = "${var.environment}_HM_AIRBYTE_DB"
  snowflake_role_name     = module.hm_airbyte_db_owner_role.name
  depends_on = [
    module.hm_airbyte_db_owner_role
  ]
}
# HM Airbyte role - owner service account
module "hm_airbyte_db_owner_service_account" {
  providers           = { snowflake = snowflake.opentofu_role }
  source              = "../../../../modules/snowflake/snowflake_user"
  snowflake_user_name = "${var.environment}_HM_AIRBYTE_DB_OWNER_SERVICE_ACCOUNT"
  default_role        = module.hm_airbyte_db_owner_role.name
  default_warehouse   = module.hm_airbyte_wh_warehouse.name
  rsa_public_key      = var.hm_airbyte_db_owner_service_account_rsa_public_key
  depends_on = [
    module.hm_airbyte_db_owner_role
  ]
}
module "grant_hm_airbyte_db_owner_role_to_hm_airbyte_db_owner_service_account" {
  providers           = { snowflake = snowflake.opentofu_role }
  source              = "../../../../modules/snowflake/snowflake_grant_role_to_user"
  snowflake_role_name = module.hm_airbyte_db_owner_role.name
  snowflake_user_name = module.hm_airbyte_db_owner_service_account.name
  depends_on = [
    module.hm_airbyte_db_owner_service_account,
    module.hm_airbyte_db_owner_role
  ]
}

# HM Kafka warehouse
module "hm_kafka_wh_warehouse" {
  providers                = { snowflake = snowflake.opentofu_role }
  source                   = "../../../../modules/snowflake/snowflake_warehouse"
  snowflake_warehouse_name = "${var.environment}_HM_KAFKA_WH"
  snowflake_warehouse_size = "xsmall"
  auto_suspend_min         = var.kafka_warehouse_auto_suspend_min
}
# HM Kafka role
locals {
  hm_kafka_db_department_names = toset([for department in var.hm_kafka_db_departments : department.name])
  hm_kafka_db_department_name_read_write_service_account_rsa_public_key_list = flatten([
    for department in var.hm_kafka_db_departments :
    {
      department_name                           = department.name
      read_write_service_account_rsa_public_key = department.read_write_service_account_rsa_public_key
    }
  ])
}
# HM Kafka role - read only role
module "hm_kafka_db_department_read_only_role" {
  providers           = { snowflake = snowflake.opentofu_role }
  source              = "../../../../modules/snowflake/snowflake_account_role"
  for_each            = local.hm_kafka_db_department_names
  snowflake_role_name = "${var.environment}_HM_KAFKA_DB_${each.value}_READ_ONLY_ROLE"
}
module "grant_database_privileges_to_hm_kafka_db_department_read_only_role" {
  providers               = { snowflake = snowflake.opentofu_role }
  source                  = "../../../../modules/snowflake/snowflake_grant_database_privileges_to_role"
  for_each                = local.hm_kafka_db_department_names
  snowflake_role_name     = "${var.environment}_HM_KAFKA_DB_${each.value}_READ_ONLY_ROLE"
  privileges              = ["USAGE"]
  snowflake_database_name = "${var.environment}_HM_KAFKA_DB"
  depends_on = [
    module.hm_kafka_db_department_read_only_role
  ]
}
module "grant_schema_privileges_to_hm_kafka_db_department_read_only_role" {
  providers               = { snowflake = snowflake.opentofu_role }
  source                  = "../../../../modules/snowflake/snowflake_grant_schema_privileges_to_role"
  for_each                = local.hm_kafka_db_department_names
  snowflake_role_name     = "${var.environment}_HM_KAFKA_DB_${each.value}_READ_ONLY_ROLE"
  privileges              = ["USAGE"]
  snowflake_database_name = "${var.environment}_HM_KAFKA_DB"
  snowflake_schema_name   = each.value
  depends_on = [
    module.hm_kafka_db_department_read_only_role
  ]
}
module "grant_warehouse_privileges_to_hm_kafka_db_department_read_only_role" {
  providers                = { snowflake = snowflake.opentofu_role }
  source                   = "../../../../modules/snowflake/snowflake_grant_warehouse_privileges_to_role"
  for_each                 = local.hm_kafka_db_department_names
  snowflake_role_name      = "${var.environment}_HM_KAFKA_DB_${each.value}_READ_ONLY_ROLE"
  privileges               = ["USAGE"]
  snowflake_warehouse_name = module.hm_kafka_wh_warehouse.name
  depends_on = [
    module.hm_kafka_db_department_read_only_role,
    module.hm_kafka_wh_warehouse
  ]
}
# HM Kafka role - read write role
module "hm_kafka_db_department_read_write_role" {
  providers           = { snowflake = snowflake.opentofu_role }
  source              = "../../../../modules/snowflake/snowflake_account_role"
  for_each            = local.hm_kafka_db_department_names
  snowflake_role_name = "${var.environment}_HM_KAFKA_DB_${each.value}_READ_WRITE_ROLE"
}
# https://docs.snowflake.com/en/user-guide/kafka-connector-install
module "grant_schema_privileges_to_hm_kafka_db_department_read_write_role" {
  providers               = { snowflake = snowflake.opentofu_role }
  source                  = "../../../../modules/snowflake/snowflake_grant_schema_privileges_to_role"
  for_each                = local.hm_kafka_db_department_names
  snowflake_role_name     = "${var.environment}_HM_KAFKA_DB_${each.value}_READ_WRITE_ROLE"
  privileges              = ["CREATE TABLE", "CREATE STAGE", "CREATE PIPE"]
  snowflake_database_name = "${var.environment}_HM_KAFKA_DB"
  snowflake_schema_name   = each.value
  depends_on = [
    module.hm_kafka_db_department_read_write_role
  ]
}
module "grant_hm_kafka_db_department_read_only_role_to_hm_kafka_db_department_read_write_role" {
  providers                    = { snowflake = snowflake.opentofu_role }
  source                       = "../../../../modules/snowflake/snowflake_grant_role_to_role"
  for_each                     = local.hm_kafka_db_department_names
  snowflake_role_name          = "${var.environment}_HM_KAFKA_DB_${each.value}_READ_ONLY_ROLE"
  snowflake_grant_to_role_name = "${var.environment}_HM_KAFKA_DB_${each.value}_READ_WRITE_ROLE"
  depends_on = [
    module.hm_kafka_db_department_read_only_role,
    module.hm_kafka_db_department_read_write_role
  ]
}
# HM Kafka role - read write service account
module "hm_kafka_db_department_read_write_service_account" {
  providers           = { snowflake = snowflake.opentofu_role }
  source              = "../../../../modules/snowflake/snowflake_user"
  for_each            = { for x in local.hm_kafka_db_department_name_read_write_service_account_rsa_public_key_list : "${x.department_name}.${x.read_write_service_account_rsa_public_key}" => x }
  snowflake_user_name = "${var.environment}_HM_KAFKA_DB_${each.value.department_name}_READ_WRITE_SERVICE_ACCOUNT"
  default_role        = "${var.environment}_HM_KAFKA_DB_${each.value.department_name}_READ_WRITE_ROLE"
  default_warehouse   = module.hm_kafka_wh_warehouse.name
  rsa_public_key      = each.value.read_write_service_account_rsa_public_key
  depends_on = [
    module.hm_kafka_db_department_read_write_role
  ]
}
module "grant_hm_kafka_db_department_read_write_role_to_hm_kafka_db_department_read_write_service_account" {
  providers           = { snowflake = snowflake.opentofu_role }
  source              = "../../../../modules/snowflake/snowflake_grant_role_to_user"
  for_each            = local.hm_kafka_db_department_names
  snowflake_role_name = "${var.environment}_HM_KAFKA_DB_${each.value}_READ_WRITE_ROLE"
  snowflake_user_name = "${var.environment}_HM_KAFKA_DB_${each.value}_READ_WRITE_SERVICE_ACCOUNT"
  depends_on = [
    module.hm_kafka_db_department_read_write_service_account,
    module.hm_kafka_db_department_read_write_role
  ]
}

# HM Streamlit warehouse
module "hm_streamlit_wh_warehouse" {
  providers                = { snowflake = snowflake.opentofu_role }
  source                   = "../../../../modules/snowflake/snowflake_warehouse"
  snowflake_warehouse_name = "${var.environment}_HM_STREAMLIT_WH"
  snowflake_warehouse_size = "xsmall"
  auto_suspend_min         = var.streamlit_warehouse_auto_suspend_min
}
# HM Streamlit role - creator role
locals {
  hm_streamlit_db_department_names = toset([for department in var.hm_streamlit_db_departments : department.name])
  hm_streamlit_db_department_name_creator_name_list = flatten([
    for department in var.hm_streamlit_db_departments :
    [
      for creator_name in department.creator_names :
      {
        department_name = department.name
        creator_name    = creator_name
      }
    ]
  ])
  hm_streamlit_db_department_name_user_name_list = flatten([
    for department in var.hm_streamlit_db_departments :
    [
      for user_name in department.user_names :
      {
        department_name = department.name
        user_name       = user_name
      }
    ]
  ])
}
module "hm_streamlit_db_department_creator_role" {
  providers           = { snowflake = snowflake.opentofu_role }
  source              = "../../../../modules/snowflake/snowflake_account_role"
  for_each            = local.hm_streamlit_db_department_names
  snowflake_role_name = "${var.environment}_HM_STREAMLIT_DB_${each.value}_CREATOR_ROLE"
}
module "grant_database_privileges_to_hm_streamlit_db_creator_role" {
  providers               = { snowflake = snowflake.opentofu_role }
  source                  = "../../../../modules/snowflake/snowflake_grant_database_privileges_to_role"
  for_each                = local.hm_streamlit_db_department_names
  snowflake_role_name     = "${var.environment}_HM_STREAMLIT_DB_${each.value}_CREATOR_ROLE"
  privileges              = ["USAGE"]
  snowflake_database_name = "${var.environment}_HM_STREAMLIT_DB"
  depends_on = [
    module.hm_streamlit_db_department_creator_role
  ]
}
module "grant_schema_privileges_to_hm_streamlit_db_creator_role" {
  providers               = { snowflake = snowflake.opentofu_role }
  source                  = "../../../../modules/snowflake/snowflake_grant_schema_privileges_to_role"
  for_each                = local.hm_streamlit_db_department_names
  snowflake_role_name     = "${var.environment}_HM_STREAMLIT_DB_${each.value}_CREATOR_ROLE"
  privileges              = ["USAGE", "CREATE STREAMLIT", "CREATE STAGE"]
  snowflake_database_name = "${var.environment}_HM_STREAMLIT_DB"
  snowflake_schema_name   = each.value
  depends_on = [
    module.hm_streamlit_db_department_creator_role
  ]
}
module "grant_hm_streamlit_db_schema_creator_role_to_user" {
  providers           = { snowflake = snowflake.opentofu_role }
  source              = "../../../../modules/snowflake/snowflake_grant_role_to_user"
  for_each            = { for x in local.hm_streamlit_db_department_name_creator_name_list : "${x.department_name}.${x.creator_name}" => x }
  snowflake_role_name = "${var.environment}_HM_STREAMLIT_DB_${each.value.department_name}_CREATOR_ROLE"
  snowflake_user_name = each.value.creator_name
  depends_on = [
    module.hm_streamlit_db_department_creator_role
  ]
}
module "grant_warehouse_privileges_to_hm_streamlit_db_department_creator_role" {
  providers                = { snowflake = snowflake.opentofu_role }
  source                   = "../../../../modules/snowflake/snowflake_grant_warehouse_privileges_to_role"
  for_each                 = local.hm_streamlit_db_department_names
  snowflake_role_name      = "${var.environment}_HM_STREAMLIT_DB_${each.value}_CREATOR_ROLE"
  privileges               = ["USAGE"]
  snowflake_warehouse_name = module.hm_streamlit_wh_warehouse.name
  depends_on = [
    module.hm_streamlit_db_department_creator_role,
    module.hm_streamlit_wh_warehouse
  ]
}
# HM Streamlit role - user role
module "hm_streamlit_db_department_user_role" {
  providers           = { snowflake = snowflake.opentofu_role }
  source              = "../../../../modules/snowflake/snowflake_account_role"
  for_each            = local.hm_streamlit_db_department_names
  snowflake_role_name = "${var.environment}_HM_STREAMLIT_DB_${each.value}_USER_ROLE"
}
module "grant_database_privileges_to_hm_streamlit_db_user_role" {
  providers               = { snowflake = snowflake.opentofu_role }
  source                  = "../../../../modules/snowflake/snowflake_grant_database_privileges_to_role"
  for_each                = local.hm_streamlit_db_department_names
  snowflake_role_name     = "${var.environment}_HM_STREAMLIT_DB_${each.value}_USER_ROLE"
  privileges              = ["USAGE"]
  snowflake_database_name = "${var.environment}_HM_STREAMLIT_DB"
  depends_on = [
    module.hm_streamlit_db_department_user_role
  ]
}
module "grant_schema_privileges_to_hm_streamlit_db_user_role" {
  providers               = { snowflake = snowflake.opentofu_role }
  source                  = "../../../../modules/snowflake/snowflake_grant_schema_privileges_to_role"
  for_each                = local.hm_streamlit_db_department_names
  snowflake_role_name     = "${var.environment}_HM_STREAMLIT_DB_${each.value}_USER_ROLE"
  privileges              = ["USAGE"]
  snowflake_database_name = "${var.environment}_HM_STREAMLIT_DB"
  snowflake_schema_name   = each.value
  depends_on = [
    module.hm_streamlit_db_department_user_role
  ]
}
module "grant_hm_streamlit_db_schema_user_role_to_user" {
  providers           = { snowflake = snowflake.opentofu_role }
  source              = "../../../../modules/snowflake/snowflake_grant_role_to_user"
  for_each            = { for x in local.hm_streamlit_db_department_name_user_name_list : "${x.department_name}.${x.user_name}" => x }
  snowflake_role_name = "${var.environment}_HM_STREAMLIT_DB_${each.value.department_name}_USER_ROLE"
  snowflake_user_name = each.value.user_name
  depends_on = [
    module.hm_streamlit_db_department_user_role
  ]
}
module "grant_warehouse_privileges_to_hm_streamlit_db_department_user_role" {
  providers                = { snowflake = snowflake.opentofu_role }
  source                   = "../../../../modules/snowflake/snowflake_grant_warehouse_privileges_to_role"
  for_each                 = local.hm_streamlit_db_department_names
  snowflake_role_name      = "${var.environment}_HM_STREAMLIT_DB_${each.value}_USER_ROLE"
  privileges               = ["USAGE"]
  snowflake_warehouse_name = module.hm_streamlit_wh_warehouse.name
  depends_on = [
    module.hm_streamlit_db_department_user_role,
    module.hm_streamlit_wh_warehouse
  ]
}
