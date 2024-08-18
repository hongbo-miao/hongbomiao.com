terraform {
  required_providers {
    kubectl = {
      source = "gavinbunney/kubectl"
    }
  }
}

data "kubectl_path_documents" "main" {
  pattern = "${var.manifest_dir_path}/*.yaml"
}
# https://registry.terraform.io/providers/gavinbunney/kubectl/latest/docs/resources/kubectl_manifest
resource "kubectl_manifest" "main" {
  count = length(
    flatten(
      toset([
        for f in fileset(".", data.kubectl_path_documents.main.pattern) : split("\n---\n", file(f))
      ])
    )
  )
  yaml_body         = element(data.kubectl_path_documents.main.documents, count.index)
  server_side_apply = true
  wait              = true
}
