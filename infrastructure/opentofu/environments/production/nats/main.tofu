
module "stream_emergency_transcriptions" {
  source            = "../../../modules/nats/jetstream_stream"
  name              = "EMERGENCY_TRANSCRIPTIONS"
  subjects          = ["EMERGENCY_TRANSCRIPTIONS.>"]
  storage           = "file"
  retention         = "limits"
  max_message_count = -1
  max_age           = 0
  max_bytes         = 500000000000 # 500 GB
  discard           = "old"
  ack               = true
  replica_count     = 3
}
