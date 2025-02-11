resource "harbor_registry" "docker_hub" {
  provider_name = "docker-hub"
  name          = "docker-hub"
  endpoint_url  = "https://hub.docker.com"
}

resource "harbor_project" "docker_hub_proxy_cache" {
  name        = "docker-hub-proxy-cache"
  public      = true
  registry_id = harbor_registry.docker_hub.registry_id
}
