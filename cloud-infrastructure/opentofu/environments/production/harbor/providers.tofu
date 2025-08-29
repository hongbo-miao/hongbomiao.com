provider "harbor" {
  url = "https://harbor.hongbomiao.com"
}

provider "aws" {
  alias  = "production"
  region = "us-west-2"
}
