# https://registry.terraform.io/providers/gr8-toolkit/ranger/latest/docs
# Talks to the Ranger Admin Tilt port-forward (see Tiltfile), not an in-cluster service name.
provider "ranger" {
  url             = "http://localhost:6080"
  username        = "admin"
  password        = "Passw0rd1"
  skip_tls_verify = true
}
