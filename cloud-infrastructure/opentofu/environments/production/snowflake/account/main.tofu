module "snowflake_account_parameter_allow_client_mfa_caching" {
  providers = { snowflake = snowflake.account_admin }
  source    = "../../../../modules/snowflake/snowflake_account_parameter"
  key       = "ALLOW_CLIENT_MFA_CACHING"
  value     = "true"
}
module "snowflake_account_parameter_enable_unredacted_query_syntax_error" {
  providers = { snowflake = snowflake.account_admin }
  source    = "../../../../modules/snowflake/snowflake_account_parameter"
  key       = "ENABLE_UNREDACTED_QUERY_SYNTAX_ERROR"
  value     = "true"
}
