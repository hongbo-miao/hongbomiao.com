output "instance_id" {
  value = aws_instance.hm_ec2_instance.id
}

output "instance_public_ip" {
  value = aws_instance.hm_ec2_instance.public_ip
}
