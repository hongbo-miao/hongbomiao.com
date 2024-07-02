terraform {
  required_providers {
    airbyte = {
      source = "airbytehq/airbyte"
    }
  }
}

# https://registry.terraform.io/providers/airbytehq/airbyte/latest/docs/resources/source_jira
resource "airbyte_source_jira" "hm_airbyte_source_jira" {
  name         = var.name
  workspace_id = var.workspace_id
  configuration = {
    domain    = var.jira_domain
    email     = var.jira_user_email
    api_token = var.jira_user_api_token
  }
}
