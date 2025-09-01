module "hm_object_storage" {
  source                = "../../../../modules/nebius/object_storage"
  project_id            = var.project_id
  object_storage_name   = "${var.environment}-hm-bucket"
  default_storage_class = "STANDARD"
  versioning_policy     = "ENABLED"
  labels                = var.common_labels
}
