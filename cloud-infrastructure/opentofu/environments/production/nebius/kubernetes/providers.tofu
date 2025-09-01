locals {
  account_id = "serviceaccount-u00bxj270cjv611fc8"
}

provider "nebius" {
  service_account = {
    account_id       = local.account_id
    public_key_id    = "publickey-u00eas1vqpv32j9pcz"
    private_key_file = "~/.nebius/${local.account_id}-credentials.pem"
  }
}
provider "helm" {
  kubernetes = {
    config_path = "~/.kube/config"
  }
}
provider "kubernetes" {
  config_path = "~/.kube/config"
}
