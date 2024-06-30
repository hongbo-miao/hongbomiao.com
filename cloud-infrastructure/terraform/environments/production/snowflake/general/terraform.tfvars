environment                                      = "production"
snowflake_public_schema_name                     = "PUBLIC"
production_department_warehouse_auto_suspend_min = 5
production_airbyte_warehouse_auto_suspend_min    = 60
production_kafka_warehouse_auto_suspend_min      = 60
production_streamlit_warehouse_auto_suspend_min  = 5
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
production_hm_airbyte_db_owner_user_rsa_public_key_without_header_and_trailer = "MIICIjANBgkqhkiG9w0BAQEFAAOCAg8AMIICCgKCAgEAv+6lly9GIIVegTeZig3k\njGbhOCp2TRsDxTWcmCD3KlZTtotWhykqnN0/jdaseFjOzT4IOx3DcVIVqMVyHZtC\nFn8w3M26JL8BClH6eIunlQiy01ZtbVoi5zQc2VHkPKj3pPwVbZ8K2aoCqw3ynHze\nVf0OZIeIIBYfuLtKeJ12Fuyj1KzzB/ePGJ1JOhH6bwvup/q6LKtt8ZZOly8oUxSu\nob6ypU+rgnyaH1R+zHhIN0legCCVZPLszi65bSZd9kYc3UZqpFFY1taUyVMKvYkF\nISrn6+rS+mUSV/zZ00rfW+GVjiqcnuNu9UT9njxtBUmM1JEfEQng05PtcvUq0SY/\nPKCwm9tjxEmXbJEzWRm59Rm44tdq4iOzMONc4ZNUoc8O8JjLdaP91fx1Q3FnTW58\n2sw3iBch/oJfabdavSQKm4LC+JnugV6ag4oJ+SlFso7TVX+lWz6j8Fv8HluEouTQ\nFpojQWH9ndEmLH/HxmPIxD9fZpkxtUmbowiRbXo1Wh644GAR9Vd6Gr/d5oX0XgZQ\n35JipKQoB+UQW/Clvz30EREVZ7v+ev162qJ3x9GGEog9wCyEkyAGyhHPzuD3d1dX\nnpZ1QsqSgQhq9HA9QDRMhgywMAqpMV/WqkYRyKSdLmtXJJFJq0qxM1kvEq/TuC4x\n0AiTOkGb26XDegmpjbLpJSUCAwEAAQ=="
