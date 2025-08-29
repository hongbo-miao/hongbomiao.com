provider "snowflake" {
  alias         = "opentofu_role"
  role          = "PRODUCTION_OPENTOFU_ROLE"
  authenticator = "SNOWFLAKE_JWT"
}
