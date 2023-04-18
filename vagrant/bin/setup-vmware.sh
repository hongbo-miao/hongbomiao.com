#!/usr/bin/env bash
set -e

# https://developer.hashicorp.com/vagrant/docs/providers/vmware/installation
brew install --cask vmware-fusion
brew install --cask vagrant-vmware-utility
vagrant plugin install vagrant-vmware-desktop
