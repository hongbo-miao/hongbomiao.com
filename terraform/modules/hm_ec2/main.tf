provider "aws" {
  region = "us-west-2"
}

resource "aws_instance" "hm_ec2_instance" {
  # Ubuntu Server 22.04 LTS (HVM), SSD Volume Type (64-bit (Arm))
  ami           = "ami-0c79a55dda52434da"
  instance_type = "t2.nano"
  tags = {
    Name = var.ec2_instance_name
  }
}
