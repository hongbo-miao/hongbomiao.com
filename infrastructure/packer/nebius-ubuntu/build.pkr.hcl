packer {
  required_plugins {
    nebius = {
      source  = "github.com/nebius/nebius"
      version = ">= 0.0.4"
    }
  }
}

source "nebius-image" "ubuntu" {
  communicator = "ssh"
  ssh_username = "ubuntu"
  service_account {
    private_key_file = pathexpand(var.service_account_private_key_file)
    public_key_id    = var.service_account_public_key_id
    account_id       = var.service_account_id
  }
  disk {
    size_gibibytes = 20
  }
  base_image {
    id = var.base_image_id
  }
  network {
    subnet_id                   = var.vpc_subnet_id
    associate_public_ip_address = true
  }
  instance {
    platform = "cpu-d3"
    preset   = "4vcpu-16gb"
  }
  image {
    name                        = "nebius-ubuntu-btop-${var.image_version}"
    version                     = var.image_version
    image_family                = "nebius-ubuntu-btop"
    image_family_human_readable = "Nebius Ubuntu with btop"
    cpu_architecture            = "amd64"
  }
  parent_id = var.project_id
}

build {
  sources = ["source.nebius-image.ubuntu"]
  provisioner "shell" {
    inline_shebang = "/bin/bash"
    inline = [
      "set -euo pipefail",

      "echo 'Installing btop'",
      "sudo apt-get update",
      "sudo apt-get install --yes btop",

      # Fail the build rather than publish an image where the baked-in tool is broken.
      "echo 'Verifying btop is installed'",
      "btop --version",

      "echo 'Resetting cloud-init and caches'",
      "sudo apt-get clean",
      "sudo cloud-init clean --logs",
      "sudo sync",
    ]
  }
}
