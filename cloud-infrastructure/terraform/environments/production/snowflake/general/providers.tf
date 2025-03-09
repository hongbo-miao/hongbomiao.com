provider "snowflake" {
  alias         = "terraform_role"
  role          = "PRODUCTION_TERRAFORM_ROLE"
  authenticator = "SNOWFLAKE_JWT"
}
