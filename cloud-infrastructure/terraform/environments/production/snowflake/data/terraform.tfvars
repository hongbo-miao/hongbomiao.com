environment                  = "PRODUCTION"
public_schema_name           = "PUBLIC"
database_data_retention_days = 90
department_db_departments = [
  {
    name = "ENGINEERING"
    schemas = [
      { name = "PUBLIC" }
    ]
  },
  {
    name = "PRODUCT"
    schemas = [
      { name = "PUBLIC" },
      { name = "TRACKER" }
    ]
  }
]
hm_streamlit_db_departments = [
  { name = "ENGINEERING" },
  { name = "PRODUCT" }
]
