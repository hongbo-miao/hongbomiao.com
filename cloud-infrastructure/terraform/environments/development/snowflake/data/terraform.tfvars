environment                              = "development"
snowflake_public_schema_name             = "PUBLIC"
development_database_data_retention_days = 30
development_department_db_departments = [
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
