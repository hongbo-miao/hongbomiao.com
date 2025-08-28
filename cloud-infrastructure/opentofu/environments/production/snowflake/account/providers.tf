provider "snowflake" {
  alias         = "account_admin"
  role          = "ACCOUNTADMIN"
  authenticator = "SNOWFLAKE_JWT"
}
