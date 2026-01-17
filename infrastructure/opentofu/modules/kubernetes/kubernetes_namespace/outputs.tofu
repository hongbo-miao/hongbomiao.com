output "namespace" {
  value = kubernetes_namespace_v1.main.metadata[0].name
}
