# Source - Postgres: production-hm-postgres | Database: iot_db | Schema: public
data "aws_secretsmanager_secret" "production_hm_postgres_airbyte_user_secret" {
  name = "production-prime-radiant-postgres/airbyte-user"
}
data "aws_secretsmanager_secret_version" "production_hm_postgres_airbyte_user_secret_version" {
  secret_id = data.aws_secretsmanager_secret.production_hm_postgres_airbyte_user_secret.id
}
module "airbyte_source_production_hm_postgres_iot_db_database_public_schema" {
  source                    = "../../../modules/airbyte/hm_airbyte_source_postgres"
  name                      = "production-hm-postgres-iot-db-database-public-schema"
  workspace_id              = var.airbyte_workspace_id
  postgres_host             = "production-hm-postgres.xxxxxxxxxxxx.us-west-2.rds.amazonaws.com"
  postgres_port             = 5432
  postgres_user_name        = jsondecode(data.aws_secretsmanager_secret_version.production_hm_postgres_airbyte_user_secret_version.secret_string)["user_name"]
  postgres_password         = jsondecode(data.aws_secretsmanager_secret_version.production_hm_postgres_airbyte_user_secret_version.secret_string)["password"]
  postgres_database         = "iot_db"
  postgres_schemas          = ["public"]
  postgres_replication_slot = "airbyte_slot"
  postgres_publication      = "airbyte_publication"
}
# Destination - Snowflake | Database: PRODUCTION_HM_AIRBYTE_DB | Schema: ENGINEERING_IOT_DB_DATABASE_PUBLIC_SCHEMA
data "aws_secretsmanager_secret" "snowflake_production_hm_airbyte_db_owner_secret" {
  name = "hm/snowflake/production_hm_airbyte_db/owner"
}
data "aws_secretsmanager_secret_version" "snowflake_production_hm_airbyte_db_owner_secret_version" {
  secret_id = data.aws_secretsmanager_secret.snowflake_production_hm_airbyte_db_owner_secret.id
}
module "airbyte_destination_snowflake_production_hm_airbyte_db_database_engineering_iot_db_database_public_schema" {
  source                                = "../../../modules/airbyte/hm_airbyte_destination_snowflake"
  name                                  = "snowflake-production-hm-airbyte-db-database-engineering-public-schema"
  workspace_id                          = var.airbyte_workspace_id
  snowflake_host                        = var.snowflake_host
  snowflake_warehouse                   = "HM_PRODUCTION_HM_AIRBYTE_WH"
  snowflake_database                    = "PRODUCTION_HM_AIRBYTE_DB"
  snowflake_schema                      = "ENGINEERING_IOT_DB_DATABASE_PUBLIC_SCHEMA"
  snowflake_role                        = "HM_PRODUCTION_HM_AIRBYTE_DB_OWNER_ROLE"
  snowflake_user_name                   = "HM_PRODUCTION_HM_AIRBYTE_DB_OWNER_USER"
  snowflake_user_private_key            = jsondecode(data.aws_secretsmanager_secret_version.snowflake_production_hm_airbyte_db_owner_secret_version.secret_string)["private_key"]
  snowflake_user_private_key_passphrase = jsondecode(data.aws_secretsmanager_secret_version.snowflake_production_hm_airbyte_db_owner_secret_version.secret_string)["private_key_passphrase"]
}
# Connection
# - Source - Postgres: production-hm-postgres | Database: iot_db | Schema: public
# - Destination - Snowflake | Database: PRODUCTION_HM_AIRBYTE_DB | Schema: ENGINEERING_IOT_DB_DATABASE_PUBLIC_SCHEMA
module "hm_airbyte_connection_production_iot_postgres_iot_db_database_public_schema_to_snowflake_production_hm_airbyte_db_database_engineering_iot_db_database_public_schema" {
  source         = "../../../modules/airbyte/hm_airbyte_connection"
  name           = "${module.airbyte_source_production_hm_postgres_iot_db_database_public_schema.name}-to-${module.airbyte_destination_snowflake_production_hm_airbyte_db_database_engineering_iot_db_database_public_schema.name}"
  source_id      = module.airbyte_source_production_hm_postgres_iot_db_database_public_schema.id
  destination_id = module.airbyte_destination_snowflake_production_hm_airbyte_db_database_engineering_iot_db_database_public_schema.id
  streams = [
    {
      name      = "users"
      sync_mode = "incremental_append"
    },
    {
      name      = "experiments"
      sync_mode = "incremental_append"
    }
  ]
  depends_on = [
    module.airbyte_source_production_hm_postgres_iot_db_database_public_schema,
    module.airbyte_destination_snowflake_production_hm_airbyte_db_database_engineering_iot_db_database_public_schema
  ]
}

# Source - CSV: cities
module "airbyte_source_csv_cities" {
  source       = "../../../modules/airbyte/hm_airbyte_source_csv"
  name         = "cities"
  workspace_id = var.airbyte_workspace_id
  dataset_name = "cities"
  url          = "https://people.sc.fsu.edu/~jburkardt/data/csv/cities.csv"
}
# Destination - Snowflake | Database: PRODUCTION_HM_AIRBYTE_DB | Schema: DATA_ENGINEERING_WORLD_SCHEMA
module "airbyte_destination_snowflake_production_hm_airbyte_db_database_data_engineering_world_schema" {
  source                                = "../../../modules/airbyte/hm_airbyte_destination_snowflake"
  name                                  = "snowflake-production-hm-airbyte-db-database-hongbo-test-cities-schema"
  workspace_id                          = var.airbyte_workspace_id
  snowflake_host                        = var.snowflake_host
  snowflake_warehouse                   = "HM_PRODUCTION_HM_AIRBYTE_WH"
  snowflake_database                    = "PRODUCTION_HM_AIRBYTE_DB"
  snowflake_schema                      = "ENGINEERING_WORLD"
  snowflake_role                        = "HM_PRODUCTION_HM_AIRBYTE_DB_OWNER_ROLE"
  snowflake_user_name                   = "HM_PRODUCTION_HM_AIRBYTE_DB_OWNER_USER"
  snowflake_user_private_key            = jsondecode(data.aws_secretsmanager_secret_version.snowflake_production_hm_airbyte_db_owner_secret_version.secret_string)["private_key"]
  snowflake_user_private_key_passphrase = jsondecode(data.aws_secretsmanager_secret_version.snowflake_production_hm_airbyte_db_owner_secret_version.secret_string)["private_key_passphrase"]
}
# Connection
# - Source - CSV: cities
# - Destination - Snowflake | Database: PRODUCTION_HM_AIRBYTE_DB | Schema: DATA_ENGINEERING_WORLD_SCHEMA
module "hm_airbyte_connection_csv_cities_to_snowflake_production_hm_airbyte_db_database_data_science_prime_radiant_database_prt_schema" {
  source         = "../../../modules/airbyte/hm_airbyte_connection"
  name           = "${module.airbyte_source_csv_cities.name}-to-${module.airbyte_destination_snowflake_production_hm_airbyte_db_database_data_engineering_world_schema.name}"
  source_id      = module.airbyte_source_csv_cities.id
  destination_id = module.airbyte_destination_snowflake_production_hm_airbyte_db_database_data_engineering_world_schema.id
  streams = [
    {
      name      = "cities"
      sync_mode = "full_refresh_overwrite"
    }
  ]
  depends_on = [
    module.airbyte_source_csv_cities,
    module.airbyte_destination_snowflake_production_hm_airbyte_db_database_data_engineering_world_schema
  ]
}
