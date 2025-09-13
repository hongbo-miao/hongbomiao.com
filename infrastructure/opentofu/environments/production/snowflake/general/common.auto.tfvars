environment                           = "PRODUCTION"
public_schema_name                    = "PUBLIC"
department_warehouse_auto_suspend_min = 5
airbyte_warehouse_auto_suspend_min    = 60
kafka_warehouse_auto_suspend_min      = 60
streamlit_warehouse_auto_suspend_min  = 5
department_db_departments = [
  {
    name             = "ENGINEERING"
    admin_user_names = ["ME@HONGBOMIAO.COM"]
    schemas = [
      { name = "PUBLIC" },
      {
        name                                      = "MOTOR"
        read_only_user_names                      = ["ME@HONGBOMIAO.COM"]
        read_write_user_names                     = ["ME@HONGBOMIAO.COM"]
        read_only_service_account_rsa_public_key  = "MIICIjANBgkqhkiG9w0BAQEFAAOCAg8AMIICCgKCAgEAr6980UHshnBMZudm5zif\nhy8KNSixz6LRudEIrd5fFU4hHRDEZGmJF1LQipVmrbkIur9Oz0xV4gge582d8jQx\nJ6SIBhzDYrKwFPD4OrJXJO5W0A2kEE6y1iJd7Jl8lofB5ocJm53ACYOD9LRYlqVu\n+orfdkEqVCi45SjOS2H+5Dgo/gtKID8EUUb/6TFhBGwXr+ZDK9fiJrGt/kUNtRCS\ntzKNiQlEzI2hyS1GaZuOL/oFrnveJHU7dzCz7OeEOhduu5BDNUeiACsHNLpTzUph\nyCyXs/DI4d1h2Ok2C/a6E6GaVdQJFyZQRWtW7pmWhcTNl+XONqz3dHSTkPO4diBy\nvXYFW9UmLGG/18UADIvU7r5J6oVSyIwgPGlRCB4IFNb7uGfXU+d6QUyFwxWHjwyq\ny+OWmmyK1wQC66aCrHOkIbn16AdAZUcf1H+qoiJUQJEYNtGwA+aJ3j0N+URKMGkF\nMyabROTO9L9v5hbr8lk193pTpn2lfC5kMjKvSgZxykSy6iMLJCHpcqcXWnpRvCUO\nEDMD+w7utg9di0/Lk9XqjLD8pPAXYqK7rP7AJGa4qZr54iw/QpaJrh1srsUGA025\nJI9VhJdd80gylhs6GuxiFBktKHa49tgZ+u7UgBOsujH1RZ7wO4I+NxBVumzPp7RN\n6LtuiCKQR3AGzVyb8O7hPjsCAwEAAQ=="
        read_write_service_account_rsa_public_key = "MIICIjANBgkqhkiG9w0BAQEFAAOCAg8AMIICCgKCAgEAufr++4kazyAcBcP2VVqe\nlk/JRSfdXE7PEucF/j5zVtNeRm56A8/1GejWFGHqdMZpSbWeaC8s7SrMImliKCe8\nwb2/U44cz2j/+IOvxKvJbWIauiXpCpw4rlEteRUIP5u//XKPh3my+2eAuwv51Up9\n6KmAPLK0OYOikDv5/ZprQClMn2vW/iizK6r1lTtrp4/hhlC1K6MCjgqbCOTohxHF\niKc0kC9ACPjszyZN1owMlosBn89JIRI2VCjU3V1eY9iRdstuX317rVDG6fVrqhP8\n00vb7n0gMZAvULdtH4MobvYHidlGcETJHtRV3PhMLT6777ZO4EDSY2mspw0Qcdyb\nyNF7aWIuP7oSn+S5rOHlWR7vle16LYZAHN3DiDMJjkqoFzc9CNS4cOvMfO8GjDbY\ntUW8Dt1fsuyxksrxo2yeapZ9AkyttZgqtiBzaeYnm5V1vEtEeXH5P6q+7U1xH67S\nbZ9070vDuLB1mC0USnxjb16DLOXh7t+uCioeXjYHc0oqeo/YJ9zq5pGtB/ocCnie\n29Qh84GFXTtStKYaQ+KL9Qi0D5xJVs8rfnf3cPj5odCS6vuhT6mAkxub2wxSZZzW\nZINQy1yIHdo+0STGTZJAXiHHzxkTlecCWPvy3OXj3ug3nv/QHIH6DbETKa1DxNIT\n0VBq4gNM8/QdQqqUMy5cKL8CAwEAAQ=="
      }
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
  {
    name          = "ENGINEERING"
    creator_names = ["ME@HONGBOMIAO.COM"]
    user_names    = ["ME@HONGBOMIAO.COM"]
  },
  { name = "PRODUCT" }
]
hm_kafka_db_departments = [
  {
    name                                      = "PRODUCT"
    read_write_service_account_rsa_public_key = "MIICIjANBgkqhkiG9w0BAQEFAAOCAg8AMIICCgKCAgEAsRSL5AF6TAuRDVIztaPD\nvQeHEdYsVVwwgGkWOT4k/MPp3ttGGbHcMIL4sIc4aNkVpHu25Y1p9FU0Msnh3SRB\nO/tyVpJV9kY5TPueWWG/ltA89lnVxPXCXfDLoIr0N/8JMkA+L7nez+d1FmUNLngG\nPRZ5GfUqHahtJrTJ9pkboM4t3sU3vJ+AKp+iGfkmpZvtjdunifmd3NdnJbPOiU1A\n6Pl7N2e7NRLJqxdo7PPmSCHTTvXFciGwyeU9dRi4KAWrv0YZJDzEjD6UPSgOxfL1\nwI5Oh4LEcXOIaPrRKa1eQNOIcqmtE5AiaYIpDyttNWi60Y4GIFyLrUKWc2ksynHr\nJenbKnDtCA2nm7/lWIUbgl2C8unDCtDv009uNaMTdck8A7owW18wKkciSet6oY3z\n3DOHLkFPCxOZdkmXttQzS1kCR6e+p8svmyBm05JnT2Ji9TzDhR2uRMZiEL4pebh0\nzmJt3NHq3jzfRxtgFbcKP6QO1AprhgLeJdKY43lxAGSrF3VJ57wWZ0Bsi7G75q9c\neWDXqOcIKGDBo7KPdlXm53bCMVuwmBpmExylkLqMVMZhklqOTaVB04pK0kd9vh1n\nPKtt6FpxS4kW3Vm5XEFG4ige+rnpwX/wzAwtbsbihpENKpmf/whDWROgQmYYNwa9\nLGh5pfQGvpI3EqnOIRg6bRMCAwEAAQ=="
  }
]
hm_airbyte_db_owner_service_account_rsa_public_key = "MIICIjANBgkqhkiG9w0BAQEFAAOCAg8AMIICCgKCAgEAv+6lly9GIIVegTeZig3k\njGbhOCp2TRsDxTWcmCD3KlZTtotWhykqnN0/jdaseFjOzT4IOx3DcVIVqMVyHZtC\nFn8w3M26JL8BClH6eIunlQiy01ZtbVoi5zQc2VHkPKj3pPwVbZ8K2aoCqw3ynHze\nVf0OZIeIIBYfuLtKeJ12Fuyj1KzzB/ePGJ1JOhH6bwvup/q6LKtt8ZZOly8oUxSu\nob6ypU+rgnyaH1R+zHhIN0legCCVZPLszi65bSZd9kYc3UZqpFFY1taUyVMKvYkF\nISrn6+rS+mUSV/zZ00rfW+GVjiqcnuNu9UT9njxtBUmM1JEfEQng05PtcvUq0SY/\nPKCwm9tjxEmXbJEzWRm59Rm44tdq4iOzMONc4ZNUoc8O8JjLdaP91fx1Q3FnTW58\n2sw3iBch/oJfabdavSQKm4LC+JnugV6ag4oJ+SlFso7TVX+lWz6j8Fv8HluEouTQ\nFpojQWH9ndEmLH/HxmPIxD9fZpkxtUmbowiRbXo1Wh644GAR9Vd6Gr/d5oX0XgZQ\n35JipKQoB+UQW/Clvz30EREVZ7v+ev162qJ3x9GGEog9wCyEkyAGyhHPzuD3d1dX\nnpZ1QsqSgQhq9HA9QDRMhgywMAqpMV/WqkYRyKSdLmtXJJFJq0qxM1kvEq/TuC4x\n0AiTOkGb26XDegmpjbLpJSUCAwEAAQ=="
