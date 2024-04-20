output "arn" {
  value = aws_msk_cluster.hm_amazon_msk_cluster.arn
}
output "bootstrap_servers" {
  value = aws_msk_cluster.hm_amazon_msk_cluster.bootstrap_brokers_sasl_iam
}
