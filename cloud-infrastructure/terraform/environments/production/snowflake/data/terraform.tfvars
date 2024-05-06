environment                             = "production"
snowflake_public_schema_name            = "PUBLIC"
production_database_data_retention_days = 90
production_department_db_departments = [
  {
    name = "ENGINEERING",
    schemas = [
      {
        name = "PUBLIC"
      }
    ]
  },
  {
    name = "PRODUCT"
    schemas = [
      {
        name = "PUBLIC"
      },
      {
        name = "TRACKER"
      }
    ]
  },
  {
    name = "FINANCE"
    schemas = [
      {
        name = "PUBLIC"
      }
    ]
  }
]
