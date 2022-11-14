output "instance_id" {
  value = aws_instance.hm_cnn_instance.id
}

output "instance_public_ip" {
  value = aws_instance.hm_cnn_instance.public_ip
}
