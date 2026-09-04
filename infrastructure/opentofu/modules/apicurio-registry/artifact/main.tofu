terraform {
  required_providers {
    apicurio = {
      source = "registry.terraform.io/tae2089/apicurio-registry"
    }
  }
}

# https://registry.terraform.io/providers/tae2089/apicurio-registry/latest/docs/resources/artifact
resource "apicurio_artifact" "main" {
  group_id    = var.group_id
  artifact_id = var.artifact_id
  type        = var.type
  content     = var.content
}

#https://registry.terraform.io/providers/tae2089/apicurio-registry/latest/docs/resources/artifact_rule
resource "apicurio_artifact_rule" "compatibility" {
  group_id    = var.group_id
  artifact_id = apicurio_artifact.main.artifact_id
  type        = "COMPATIBILITY"
  config      = "FULL_TRANSITIVE"
}
