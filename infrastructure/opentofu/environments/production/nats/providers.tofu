provider "jetstream" {
  servers = "nats://${var.nats_token}@nats.hongbomiao.com:4222"
}
