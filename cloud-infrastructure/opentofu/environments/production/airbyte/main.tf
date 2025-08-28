locals {
  # https://www.javainuse.com/cron
  engineering_iot_public_airbyte_connection_schedule_cron_expression = "0 11 * * * ? * US/Pacific"      # every 1 hour at minute 11
  engineering_jira_airbyte_connection_schedule_cron_expression       = "0 32 7-19/2 ? * 2-6 US/Pacific" # every 2 hours at minute 32, 7am - 7pm, Monday - Friday
  manufacturing_iot_dbo_airbyte_connection_schedule_cron_expression  = "0 46 * * * ? * US/Pacific"      # every 1 hour at minute 46
}

# Snowflake
data "aws_secretsmanager_secret" "snowflake_hm_airbyte_db_owner_secret" {
  name = "hm/snowflake/hm_airbyte_db/owner"
}
data "aws_secretsmanager_secret_version" "snowflake_hm_airbyte_db_owner_secret_version" {
  secret_id = data.aws_secretsmanager_secret.snowflake_hm_airbyte_db_owner_secret.id
}

# Source - Postgres: production-hm-postgres | Database: iot_db | Schema: motor
data "aws_secretsmanager_secret" "hm_postgres_airbyte_user_secret" {
  name = "${var.environment}-hm-postgres/airbyte-user"
}
data "aws_secretsmanager_secret_version" "hm_postgres_airbyte_user_secret_version" {
  secret_id = data.aws_secretsmanager_secret.hm_postgres_airbyte_user_secret.id
}
module "airbyte_source_hm_postgres_iot_db_database_motor_schema" {
  source                 = "../../../modules/airbyte/hm_airbyte_source_postgres"
  name                   = "${var.environment}-hm-postgres-iot-db-motor"
  workspace_id           = var.airbyte_workspace_id
  postgres_host          = "${var.environment}-hm-postgres.xxxxxxxxxxxx.us-west-2.rds.amazonaws.com"
  postgres_port          = 5432
  postgres_user_name     = jsondecode(data.aws_secretsmanager_secret_version.hm_postgres_airbyte_user_secret_version.secret_string)["user_name"]
  postgres_password      = jsondecode(data.aws_secretsmanager_secret_version.hm_postgres_airbyte_user_secret_version.secret_string)["password"]
  postgres_database      = "iot_db"
  postgres_schema        = "public"
  initial_waiting_time_s = 120
  # tunnel_method = {
  #   ssh_key_authentication = {
  #     tunnel_host = "xxx.xxx.xxx.xxx"
  #     tunnel_port = 22
  #     tunnel_user = "ubuntu"
  #     ssh_key     = jsondecode(data.aws_secretsmanager_secret_version.hm_postgres_airbyte_user_secret_version.secret_string)["tunnel_ssh_private_key"]
  #   }
  # }
}
# Destination - Snowflake | Database: PRODUCTION_HM_AIRBYTE_DB | Schema: ENGINEERING_IOT_DB_MOTOR
module "airbyte_destination_snowflake_hm_airbyte_db_database_engineering_iot_db_database_motor_schema" {
  source                                = "../../../modules/airbyte/hm_airbyte_destination_snowflake"
  name                                  = "${var.environment}-engineering-motor"
  workspace_id                          = var.airbyte_workspace_id
  snowflake_host                        = var.snowflake_host
  snowflake_warehouse                   = "${upper(var.environment)}_HM_AIRBYTE_WH"
  snowflake_database                    = "${upper(var.environment)}_HM_AIRBYTE_DB"
  snowflake_schema                      = "ENGINEERING_IOT_DB_MOTOR"
  snowflake_role                        = "${upper(var.environment)}_HM_AIRBYTE_DB_OWNER_ROLE"
  snowflake_user_name                   = "${upper(var.environment)}_HM_AIRBYTE_DB_OWNER_SERVICE_ACCOUNT"
  snowflake_user_private_key            = jsondecode(data.aws_secretsmanager_secret_version.snowflake_hm_airbyte_db_owner_secret_version.secret_string)["private_key"]
  snowflake_user_private_key_passphrase = jsondecode(data.aws_secretsmanager_secret_version.snowflake_hm_airbyte_db_owner_secret_version.secret_string)["private_key_passphrase"]
}
# Connection
# - Source - Postgres: production-hm-postgres | Database: iot_db | Schema: motor
# - Destination - Snowflake | Database: PRODUCTION_HM_AIRBYTE_DB | Schema: ENGINEERING_IOT_DB_MOTOR
module "airbyte_connection_snowflake_hm_airbyte_db_database_engineering_iot_db_database_motor_schema" {
  source                               = "../../../modules/airbyte/hm_airbyte_connection"
  source_id                            = module.airbyte_source_hm_postgres_iot_db_database_motor_schema.id
  destination_id                       = module.airbyte_destination_snowflake_hm_airbyte_db_database_engineering_iot_db_database_motor_schema.id
  destination_name                     = module.airbyte_destination_snowflake_hm_airbyte_db_database_engineering_iot_db_database_motor_schema.name
  schedule_type                        = "cron"
  schedule_cron_expression             = local.engineering_iot_public_airbyte_connection_schedule_cron_expression
  non_breaking_schema_updates_behavior = "propagate_fully"
  status                               = "active"
  streams = [
    {
      name      = "_airbyte_heartbeat"
      sync_mode = "full_refresh_overwrite"
    },
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
    module.airbyte_source_hm_postgres_iot_db_database_motor_schema,
    module.airbyte_destination_snowflake_hm_airbyte_db_database_engineering_iot_db_database_motor_schema
  ]
}

# Source - Microsoft SQL Server: production-hm-microsoft-sql-server | Database: iot_db | Schema: dbo
data "aws_secretsmanager_secret" "production_manufacturing_iot_airbyte_user_secret" {
  name = "production-manufacturing-iot/airbyte-user"
}
data "aws_secretsmanager_secret_version" "production_manufacturing_iot_airbyte_user_secret_version" {
  secret_id = data.aws_secretsmanager_secret.production_manufacturing_iot_airbyte_user_secret.id
}
module "manufacturing_airbyte_source_production_manufacturing_iot_db_database_dbo_schema" {
  source                         = "../../../modules/airbyte/hm_airbyte_source_microsoft_sql_server"
  name                           = "production-manufacturing-iot-dbo"
  workspace_id                   = var.airbyte_workspace_id
  microsoft_sql_server_host      = "${var.environment}-hm-microsoft-sql-server.xxxxxxxxxxxx.us-west-2.rds.amazonaws.com"
  microsoft_sql_server_port      = 1433
  microsoft_sql_server_database  = "iot_db"
  microsoft_sql_server_schema    = "dbo"
  initial_waiting_time_s         = 300
  microsoft_sql_server_user_name = jsondecode(data.aws_secretsmanager_secret_version.production_manufacturing_iot_airbyte_user_secret_version.secret_string)["user_name"]
  microsoft_sql_server_password  = jsondecode(data.aws_secretsmanager_secret_version.production_manufacturing_iot_airbyte_user_secret_version.secret_string)["password"]
}
# Destination - Snowflake | Database: PRODUCTION_HM_AIRBYTE_DB | Schema: MANUFACTURING_IOT_DB_DBO
module "hm_airbyte_destination_snowflake_production_hm_airbyte_db_database_manufacturing_iot_dbo_schema" {
  source                                = "../../../modules/airbyte/hm_airbyte_destination_snowflake"
  name                                  = "production-manufacturing-iot-dbo"
  workspace_id                          = var.airbyte_workspace_id
  snowflake_host                        = var.snowflake_host
  snowflake_warehouse                   = "${upper(var.environment)}_HM_AIRBYTE_WH"
  snowflake_database                    = "${upper(var.environment)}_HM_AIRBYTE_DB"
  snowflake_schema                      = "MANUFACTURING_IOT_DB_DBO"
  snowflake_role                        = "${upper(var.environment)}_HM_AIRBYTE_DB_OWNER_ROLE"
  snowflake_user_name                   = "${upper(var.environment)}_HM_AIRBYTE_DB_OWNER_USER"
  snowflake_user_private_key            = jsondecode(data.aws_secretsmanager_secret_version.snowflake_hm_airbyte_db_owner_secret_version.secret_string)["private_key"]
  snowflake_user_private_key_passphrase = jsondecode(data.aws_secretsmanager_secret_version.snowflake_hm_airbyte_db_owner_secret_version.secret_string)["private_key_passphrase"]
}
# Connection
# - Source - Microsoft SQL Server: production-hm-microsoft-sql-server | Database: iot_db | Schema: dbo
# - Destination - Snowflake | Database: PRODUCTION_HM_AIRBYTE_DB | Schema: MANUFACTURING_IOT_DB_DBO
module "hm_airbyte_connection_snowflake_production_hm_airbyte_db_database_manufacturing_iot_dbo_schema" {
  source                               = "../../../modules/airbyte/hm_airbyte_connection"
  source_id                            = module.manufacturing_airbyte_source_production_manufacturing_iot_db_database_dbo_schema.id
  destination_id                       = module.hm_airbyte_destination_snowflake_production_hm_airbyte_db_database_manufacturing_iot_dbo_schema.id
  destination_name                     = module.hm_airbyte_destination_snowflake_production_hm_airbyte_db_database_manufacturing_iot_dbo_schema.name
  schedule_type                        = "cron"
  schedule_cron_expression             = local.manufacturing_iot_dbo_airbyte_connection_schedule_cron_expression
  non_breaking_schema_updates_behavior = "propagate_fully"
  status                               = "active"
  streams = [
    {
      name      = "battery"
      sync_mode = "incremental_deduped_history"
    },
    {
      name      = "motor"
      sync_mode = "incremental_deduped_history"
    }
  ]
  depends_on = [
    module.manufacturing_airbyte_source_production_manufacturing_iot_db_database_dbo_schema,
    module.hm_airbyte_destination_snowflake_production_hm_airbyte_db_database_manufacturing_iot_dbo_schema
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
# Destination - Snowflake | Database: PRODUCTION_HM_AIRBYTE_DB | Schema: ENGINEERING_WORLD
module "airbyte_destination_snowflake_hm_airbyte_db_database_engineering_world_schema" {
  source                                = "../../../modules/airbyte/hm_airbyte_destination_snowflake"
  name                                  = "${var.environment}-engineering-world-cities"
  workspace_id                          = var.airbyte_workspace_id
  snowflake_host                        = var.snowflake_host
  snowflake_warehouse                   = "${upper(var.environment)}_HM_AIRBYTE_WH"
  snowflake_database                    = "${upper(var.environment)}_HM_AIRBYTE_DB"
  snowflake_schema                      = "ENGINEERING_WORLD"
  snowflake_role                        = "${upper(var.environment)}_HM_AIRBYTE_DB_OWNER_ROLE"
  snowflake_user_name                   = "${upper(var.environment)}_HM_AIRBYTE_DB_OWNER_SERVICE_ACCOUNT"
  snowflake_user_private_key            = jsondecode(data.aws_secretsmanager_secret_version.snowflake_hm_airbyte_db_owner_secret_version.secret_string)["private_key"]
  snowflake_user_private_key_passphrase = jsondecode(data.aws_secretsmanager_secret_version.snowflake_hm_airbyte_db_owner_secret_version.secret_string)["private_key_passphrase"]
}
# Connection
# - Source - CSV: cities
# - Destination - Snowflake | Database: PRODUCTION_HM_AIRBYTE_DB | Schema: ENGINEERING_WORLD
module "airbyte_connection_snowflake_hm_airbyte_db_database_engineering_world_schema" {
  source                               = "../../../modules/airbyte/hm_airbyte_connection"
  source_id                            = module.airbyte_source_csv_cities.id
  destination_id                       = module.airbyte_destination_snowflake_hm_airbyte_db_database_engineering_world_schema.id
  destination_name                     = module.airbyte_destination_snowflake_hm_airbyte_db_database_engineering_world_schema.name
  schedule_type                        = "manual"
  non_breaking_schema_updates_behavior = "propagate_fully"
  status                               = "active"
  streams = [
    {
      name      = "cities"
      sync_mode = "full_refresh_overwrite"
    }
  ]
  depends_on = [
    module.airbyte_source_csv_cities,
    module.airbyte_destination_snowflake_hm_airbyte_db_database_engineering_world_schema
  ]
}

# Source - Jira
data "aws_secretsmanager_secret" "hm_jira_secret" {
  name = "hm-jira"
}
data "aws_secretsmanager_secret_version" "hm_jira_secret_version" {
  secret_id = data.aws_secretsmanager_secret.hm_jira_secret.id
}
module "airbyte_source_jira" {
  source              = "../../../modules/airbyte/hm_airbyte_source_jira"
  name                = "jira"
  workspace_id        = var.airbyte_workspace_id
  jira_domain         = "hongbomiao.atlassian.net"
  jira_user_email     = jsondecode(data.aws_secretsmanager_secret_version.hm_jira_secret_version.secret_string)["user_email"]
  jira_user_api_token = jsondecode(data.aws_secretsmanager_secret_version.hm_jira_secret_version.secret_string)["user_api_token"]
}
# Destination - Snowflake | Database: PRODUCTION_HM_AIRBYTE_DB | Schema: ENGINEERING_JIRA
module "airbyte_destination_snowflake_hm_airbyte_db_database_engineering_jira_schema" {
  source                                = "../../../modules/airbyte/hm_airbyte_destination_snowflake"
  name                                  = "${var.environment}-engineering-jira"
  workspace_id                          = var.airbyte_workspace_id
  snowflake_host                        = var.snowflake_host
  snowflake_warehouse                   = "${upper(var.environment)}_HM_AIRBYTE_WH"
  snowflake_database                    = "${upper(var.environment)}_HM_AIRBYTE_DB"
  snowflake_schema                      = "ENGINEERING_JIRA"
  snowflake_role                        = "${upper(var.environment)}_HM_AIRBYTE_DB_OWNER_ROLE"
  snowflake_user_name                   = "${upper(var.environment)}_HM_AIRBYTE_DB_OWNER_SERVICE_ACCOUNT"
  snowflake_user_private_key            = jsondecode(data.aws_secretsmanager_secret_version.snowflake_hm_airbyte_db_owner_secret_version.secret_string)["private_key"]
  snowflake_user_private_key_passphrase = jsondecode(data.aws_secretsmanager_secret_version.snowflake_hm_airbyte_db_owner_secret_version.secret_string)["private_key_passphrase"]
}
# Connection
# - Source - Jira
# - Destination - Snowflake | Database: PRODUCTION_HM_AIRBYTE_DB | Schema: JIRA
module "airbyte_connection_snowflake_hm_airbyte_db_database_engineering_jira_schema" {
  source                               = "../../../modules/airbyte/hm_airbyte_connection"
  source_id                            = module.airbyte_source_jira.id
  destination_id                       = module.airbyte_destination_snowflake_hm_airbyte_db_database_engineering_jira_schema.id
  destination_name                     = module.airbyte_destination_snowflake_hm_airbyte_db_database_engineering_jira_schema.name
  schedule_type                        = "cron"
  schedule_cron_expression             = local.engineering_jira_airbyte_connection_schedule_cron_expression
  non_breaking_schema_updates_behavior = "propagate_fully"
  status                               = "active"
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
      name      = "sprints"
      sync_mode = "full_refresh_overwrite"
    },
    {
      name      = "users"
      sync_mode = "full_refresh_overwrite"
    }
  ]
  depends_on = [
    module.airbyte_source_jira,
    module.airbyte_destination_snowflake_hm_airbyte_db_database_engineering_jira_schema
  ]
}
