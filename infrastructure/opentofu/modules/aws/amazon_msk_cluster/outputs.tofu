output "arn" {
  value = aws_msk_cluster.main.arn
}
output "bootstrap_servers" {
  value = aws_msk_cluster.main.bootstrap_brokers_sasl_iam
}
