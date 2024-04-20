environment                  = "development"
snowflake_public_schema_name = "PUBLIC"
snowflake_sysadmin           = "SYSADMIN"
hongbomiao_departments = [
  {
    name             = "ENGINEERING",
    admin_user_names = ["ME@HONGBOMIAO.COM"]
    schemas = [
      {
        name                  = "PUBLIC"
        read_only_user_names  = [],
        read_write_user_names = []
      }
    ]
  },
  {
    name             = "PRODUCT"
    admin_user_names = []
    schemas = [
      {
        name                  = "PUBLIC"
        read_only_user_names  = ["ME@HONGBOMIAO.COM"],
        read_write_user_names = []
      },
      {
        name                  = "TRACKER"
        read_only_user_names  = [],
        read_write_user_names = ["ME@HONGBOMIAO.COM"]
      }
    ]
  },
  {
    name             = "FINANCE"
    admin_user_names = []
    schemas = [
      {
        name                  = "PUBLIC"
        read_only_user_names  = ["ME@HONGBOMIAO.COM"],
        read_write_user_names = []
      }
    ]
  }
]
warehouse_auto_suspend_min                                                                = 2
database_data_retention_days                                                              = 30
development_hm_kafka_db_product_read_write_user_rsa_public_key_without_header_and_trailer = "MIICIjANBgkqhkiG9w0BAQEFAAOCAg8AMIICCgKCAgEA4BfWBbPAOYP70FjWn3gW\n+qwexwPAygEjbAB/DI706LftwbgNlvuw79Rei13j7qFJpHO6jWxCjKXAPz3+qM9v\nS0zh7RYpY6mUj/UcarqSGW5E78d/XVhvomYi4JrGRHKWGV7kDN3cOpUJoHV6Idv6\n+mOpL62uRhkQtu8nzFDk0r89WgXwfUOmkvSEQvDo+oNm/wHoYt3G2AuwFa1+Ttpi\n3WBYfzPdFmDdn6wU+XeGCHo8MCeVDmVJ8TJV+g2C1cs+Ge+otCnCuCgomcvC6bGh\nhmyYcf7bZJ964hEleIjy9YYJTfyTw+/4gPuQa1APmrSY4JhlyG+IedxuRtqZSljk\nDuTfmyRCAEFrvsxJIrCPvF2EvTqCgD0uqG34JgxlHOtNtoh3PenkBD1L50ZFjjyD\noxI5aI3gyAXAfyiyFEqHN34RaRPu8gePTlHkISo/MaOtaQxenAPDt+VTpI9qqcbq\nmqx/jM7CvI+Qi93IOZ2hlFrRfYM185ui5ogXE2iXWDgSLQJIFIrdrB2EFIdpWcQE\n8DAq6Kwgt3Qie3x7C1/WCylkxtNBWz5drdVEK6ijQu4SGKtO9h+VFsrjXqwgSj77\ncvszMtdAN6wqMyXoqC7Re010ljZjxf/Ek00Dg8LdtBgxYnvWJXakZmbCyLjyQn7K\nyq+ONhh/eBYXBeagFg2j4ocCAwEAAQ=="
