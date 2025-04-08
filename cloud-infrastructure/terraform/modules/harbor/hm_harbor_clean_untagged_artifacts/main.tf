terraform {
  required_providers {
    harbor = {
      source = "goharbor/harbor"
    }
  }
}

# https://registry.terraform.io/providers/goharbor/harbor/latest/docs/resources/retention_policy
resource "harbor_retention_policy" "clean_untagged_artifacts" {
  scope    = var.project_id
  schedule = var.schedule
  rule {
    n_days_since_last_pull = var.days_since_last_pull
    repo_matching          = "**"
    tag_excluding          = "**"
    untagged_artifacts     = true
  }
}
