# Source - Postgres: production-hm-postgres | Database: iot_db | Schema: motor
data "aws_secretsmanager_secret" "production_hm_postgres_airbyte_user_secret" {
  name = "production-hm-postgres/airbyte-user"
}
data "aws_secretsmanager_secret_version" "production_hm_postgres_airbyte_user_secret_version" {
  secret_id = data.aws_secretsmanager_secret.production_hm_postgres_airbyte_user_secret.id
}
module "hm_airbyte_source_production_hm_postgres_iot_db_database_motor_schema" {
  source                    = "../../../modules/airbyte/hm_airbyte_source_postgres"
  name                      = "production-hm-postgres-iot-db-motor"
  workspace_id              = var.airbyte_workspace_id
  postgres_host             = "production-hm-postgres.xxxxxxxxxxxx.us-west-2.rds.amazonaws.com"
  postgres_port             = 5432
  postgres_user_name        = jsondecode(data.aws_secretsmanager_secret_version.production_hm_postgres_airbyte_user_secret_version.secret_string)["user_name"]
  postgres_password         = jsondecode(data.aws_secretsmanager_secret_version.production_hm_postgres_airbyte_user_secret_version.secret_string)["password"]
  postgres_database         = "iot_db"
  postgres_schemas          = ["motor"]
  postgres_replication_slot = "airbyte_slot"
  postgres_publication      = "airbyte_publication"
}
# Destination - Snowflake | Database: PRODUCTION_HM_AIRBYTE_DB | Schema: ENGINEERING_IOT_DB_MOTOR
data "aws_secretsmanager_secret" "snowflake_production_hm_airbyte_db_owner_secret" {
  name = "hm/snowflake/production_hm_airbyte_db/owner"
}
data "aws_secretsmanager_secret_version" "snowflake_production_hm_airbyte_db_owner_secret_version" {
  secret_id = data.aws_secretsmanager_secret.snowflake_production_hm_airbyte_db_owner_secret.id
}
module "hm_airbyte_destination_snowflake_production_hm_airbyte_db_database_engineering_iot_db_database_motor_schema" {
  source                                = "../../../modules/airbyte/hm_airbyte_destination_snowflake"
  name                                  = "production-engineering-motor"
  workspace_id                          = var.airbyte_workspace_id
  snowflake_host                        = var.snowflake_host
  snowflake_warehouse                   = "HM_PRODUCTION_HM_AIRBYTE_WH"
  snowflake_database                    = "PRODUCTION_HM_AIRBYTE_DB"
  snowflake_schema                      = "ENGINEERING_IOT_DB_MOTOR"
  snowflake_role                        = "HM_PRODUCTION_HM_AIRBYTE_DB_OWNER_ROLE"
  snowflake_user_name                   = "HM_PRODUCTION_HM_AIRBYTE_DB_OWNER_USER"
  snowflake_user_private_key            = jsondecode(data.aws_secretsmanager_secret_version.snowflake_production_hm_airbyte_db_owner_secret_version.secret_string)["private_key"]
  snowflake_user_private_key_passphrase = jsondecode(data.aws_secretsmanager_secret_version.snowflake_production_hm_airbyte_db_owner_secret_version.secret_string)["private_key_passphrase"]
}
# Connection
# - Source - Postgres: production-hm-postgres | Database: iot_db | Schema: motor
# - Destination - Snowflake | Database: PRODUCTION_HM_AIRBYTE_DB | Schema: ENGINEERING_IOT_DB_MOTOR
module "hm_airbyte_connection_snowflake_production_hm_airbyte_db_database_engineering_iot_db_database_motor_schema" {
  source           = "../../../modules/airbyte/hm_airbyte_connection"
  source_id        = module.hm_airbyte_source_production_hm_postgres_iot_db_database_motor_schema.id
  destination_id   = module.hm_airbyte_destination_snowflake_production_hm_airbyte_db_database_engineering_iot_db_database_motor_schema.id
  destination_name = module.hm_airbyte_destination_snowflake_production_hm_airbyte_db_database_engineering_iot_db_database_motor_schema.name
  schedule_type    = "cron"
  # https://www.javainuse.com/cron
  schedule_cron_expression = "0 0 * * * ? * US/Pacific" # every hour
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
    module.hm_airbyte_source_production_hm_postgres_iot_db_database_motor_schema,
    module.hm_airbyte_destination_snowflake_production_hm_airbyte_db_database_engineering_iot_db_database_motor_schema
  ]
}

# Source - CSV: cities
module "hm_airbyte_source_csv_cities" {
  source       = "../../../modules/airbyte/hm_airbyte_source_csv"
  name         = "cities"
  workspace_id = var.airbyte_workspace_id
  dataset_name = "cities"
  url          = "https://people.sc.fsu.edu/~jburkardt/data/csv/cities.csv"
}
# Destination - Snowflake | Database: PRODUCTION_HM_AIRBYTE_DB | Schema: ENGINEERING_WORLD
module "hm_airbyte_destination_snowflake_production_hm_airbyte_db_database_engineering_world_schema" {
  source                                = "../../../modules/airbyte/hm_airbyte_destination_snowflake"
  name                                  = "production-engineering-world-cities"
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
# - Destination - Snowflake | Database: PRODUCTION_HM_AIRBYTE_DB | Schema: ENGINEERING_WORLD
module "hm_airbyte_connection_snowflake_production_hm_airbyte_db_database_engineering_world_schema" {
  source           = "../../../modules/airbyte/hm_airbyte_connection"
  source_id        = module.hm_airbyte_source_csv_cities.id
  destination_id   = module.hm_airbyte_destination_snowflake_production_hm_airbyte_db_database_engineering_world_schema.id
  destination_name = module.hm_airbyte_destination_snowflake_production_hm_airbyte_db_database_engineering_world_schema.name
  schedule_type    = "manual"
  streams = [
    {
      name      = "cities"
      sync_mode = "full_refresh_overwrite"
    }
  ]
  depends_on = [
    module.hm_airbyte_source_csv_cities,
    module.hm_airbyte_destination_snowflake_production_hm_airbyte_db_database_engineering_world_schema
  ]
}

# Source - Jira
data "aws_secretsmanager_secret" "hm_jira_secret" {
  name = "hm-jira"
}
data "aws_secretsmanager_secret_version" "hm_jira_secret_version" {
  secret_id = data.aws_secretsmanager_secret.hm_jira_secret.id
}
module "hm_airbyte_source_jira" {
  source              = "../../../modules/airbyte/hm_airbyte_source_jira"
  name                = "jira"
  workspace_id        = var.airbyte_workspace_id
  jira_domain         = "hongbomiao.atlassian.net"
  jira_user_email     = jsondecode(data.aws_secretsmanager_secret_version.hm_jira_secret_version.secret_string)["user_email"]
  jira_user_api_token = jsondecode(data.aws_secretsmanager_secret_version.hm_jira_secret_version.secret_string)["user_api_token"]
}
# Destination - Snowflake | Database: PRODUCTION_HM_AIRBYTE_DB | Schema: ENGINEERING_JIRA
module "hm_airbyte_destination_snowflake_production_hm_airbyte_db_database_engineering_jira_schema" {
  source                                = "../../../modules/airbyte/hm_airbyte_destination_snowflake"
  name                                  = "production-engineering-jira"
  workspace_id                          = var.airbyte_workspace_id
  snowflake_host                        = var.snowflake_host
  snowflake_warehouse                   = "HM_PRODUCTION_HM_AIRBYTE_WH"
  snowflake_database                    = "PRODUCTION_HM_AIRBYTE_DB"
  snowflake_schema                      = "ENGINEERING_JIRA"
  snowflake_role                        = "HM_PRODUCTION_HM_AIRBYTE_DB_OWNER_ROLE"
  snowflake_user_name                   = "HM_PRODUCTION_HM_AIRBYTE_DB_OWNER_USER"
  snowflake_user_private_key            = jsondecode(data.aws_secretsmanager_secret_version.snowflake_production_hm_airbyte_db_owner_secret_version.secret_string)["private_key"]
  snowflake_user_private_key_passphrase = jsondecode(data.aws_secretsmanager_secret_version.snowflake_production_hm_airbyte_db_owner_secret_version.secret_string)["private_key_passphrase"]
}
# Connection
# - Source - Jira
# - Destination - Snowflake | Database: PRODUCTION_HM_AIRBYTE_DB | Schema: JIRA
module "hm_airbyte_connection_snowflake_production_hm_airbyte_db_database_engineering_jira_schema" {
  source           = "../../../modules/airbyte/hm_airbyte_connection"
  source_id        = module.hm_airbyte_source_jira.id
  destination_id   = module.hm_airbyte_destination_snowflake_production_hm_airbyte_db_database_engineering_jira_schema.id
  destination_name = module.hm_airbyte_destination_snowflake_production_hm_airbyte_db_database_engineering_jira_schema.name
  schedule_type    = "cron"
  # https://www.javainuse.com/cron
  schedule_cron_expression = "0 0 * * * ? * US/Pacific" # every hour
  streams = [
    {
      name      = "issues"
      sync_mode = "incremental_deduped_history"
    },
    {
      name      = "projects"
      sync_mode = "full_refresh_overwrite"
    },
    {
      name      = "users"
      sync_mode = "full_refresh_overwrite"
    }
  ]
  depends_on = [
    module.hm_airbyte_source_jira,
    module.hm_airbyte_destination_snowflake_production_hm_airbyte_db_database_engineering_jira_schema
  ]
}
