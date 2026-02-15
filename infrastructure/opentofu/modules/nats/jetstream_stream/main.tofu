terraform {
  required_providers {
    jetstream = {
      source = "nats-io/jetstream"
    }
  }
}

# https://search.opentofu.org/provider/nats-io/jetstream/latest/docs/resources/jetstream_stream
resource "jetstream_stream" "main" {
  name      = var.name
  subjects  = var.subjects
  storage   = var.storage
  retention = var.retention
  max_msgs  = var.max_message_count
  max_age   = var.max_age
  max_bytes = var.max_bytes
  discard   = var.discard
  replicas  = var.replica_count
}
