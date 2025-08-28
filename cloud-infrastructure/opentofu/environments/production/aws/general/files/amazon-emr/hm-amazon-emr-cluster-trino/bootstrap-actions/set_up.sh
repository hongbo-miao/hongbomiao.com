#!/usr/bin/env bash
set -e

echo "# Add SSH keys"
# An empty line is necessary since the original public key lacks a newline character at the end
{
  echo
  echo "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIPWhR5NV13iw0X8lKqsrSRqbcIJcA5AVMjyfJjOrplwH hongbo-miao"
} >> /home/hadoop/.ssh/authorized_keys
