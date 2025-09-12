locals {
  account_id = "serviceaccount-xxxxxxxxxxxxxxxxxx"
}

provider "nebius" {
  service_account = {
    account_id       = local.account_id
    public_key_id    = "publickey-xxxxxxxxxxxxxxxxxx"
    private_key_file = "~/.nebius/${local.account_id}-credentials.pem"
  }
}
