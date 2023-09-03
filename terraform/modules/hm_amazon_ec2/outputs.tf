output "ec2_instance_id" {
  value = aws_instance.hm_ec2_instance.id
}

output "ec2_instance_public_ip" {
  value = aws_instance.hm_ec2_instance.public_ip
}
