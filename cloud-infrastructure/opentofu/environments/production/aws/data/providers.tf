provider "aws" {
  alias  = "production"
  region = "us-west-2"
}
provider "helm" {
  kubernetes = {
    config_path = "~/.kube/config"
  }
}
provider "kubernetes" {
  config_path = "~/.kube/config"
}
