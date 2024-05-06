environment                           = "production"
snowflake_public_schema_name          = "PUBLIC"
production_warehouse_auto_suspend_min = 5
hongbomiao_departments = [
  {
    name             = "ENGINEERING",
    admin_user_names = []
    schemas = [
      {
        name                  = "PUBLIC"
        read_only_user_names  = ["ME@HONGBOMIAO.COM"],
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
        read_only_user_names  = ["ME@HONGBOMIAO.COM"],
        read_write_user_names = []
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
production_hm_kafka_db_departments = [
  {
    name                                                      = "PRODUCT"
    read_write_user_rsa_public_key_without_header_and_trailer = "MIICIjANBgkqhkiG9w0BAQEFAAOCAg8AMIICCgKCAgEAsRSL5AF6TAuRDVIztaPD\nvQeHEdYsVVwwgGkWOT4k/MPp3ttGGbHcMIL4sIc4aNkVpHu25Y1p9FU0Msnh3SRB\nO/tyVpJV9kY5TPueWWG/ltA89lnVxPXCXfDLoIr0N/8JMkA+L7nez+d1FmUNLngG\nPRZ5GfUqHahtJrTJ9pkboM4t3sU3vJ+AKp+iGfkmpZvtjdunifmd3NdnJbPOiU1A\n6Pl7N2e7NRLJqxdo7PPmSCHTTvXFciGwyeU9dRi4KAWrv0YZJDzEjD6UPSgOxfL1\nwI5Oh4LEcXOIaPrRKa1eQNOIcqmtE5AiaYIpDyttNWi60Y4GIFyLrUKWc2ksynHr\nJenbKnDtCA2nm7/lWIUbgl2C8unDCtDv009uNaMTdck8A7owW18wKkciSet6oY3z\n3DOHLkFPCxOZdkmXttQzS1kCR6e+p8svmyBm05JnT2Ji9TzDhR2uRMZiEL4pebh0\nzmJt3NHq3jzfRxtgFbcKP6QO1AprhgLeJdKY43lxAGSrF3VJ57wWZ0Bsi7G75q9c\neWDXqOcIKGDBo7KPdlXm53bCMVuwmBpmExylkLqMVMZhklqOTaVB04pK0kd9vh1n\nPKtt6FpxS4kW3Vm5XEFG4ige+rnpwX/wzAwtbsbihpENKpmf/whDWROgQmYYNwa9\nLGh5pfQGvpI3EqnOIRg6bRMCAwEAAQ=="
  }
]
