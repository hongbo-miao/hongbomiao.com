
module "stream_emergency_transcriptions" {
  source            = "../../../modules/nats/jetstream_stream"
  name              = "EMERGENCY_TRANSCRIPTIONS"
  subjects          = ["EMERGENCY_TRANSCRIPTIONS.>"]
  storage           = "file"
  retention         = "limits"
  max_message_count = -1
  max_age           = 0
  max_bytes         = 10737418240 # 10 GiB
  discard           = "old"
  replica_count     = 3
}
