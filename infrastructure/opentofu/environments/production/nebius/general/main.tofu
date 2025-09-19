data "nebius_iam_v1_group" "viewers_group" {
  name      = "viewers"
  parent_id = var.tenant_id
}
data "nebius_iam_v1_group" "editors_group" {
  name      = "editors"
  parent_id = var.tenant_id
}

# Rclone read-only service account
module "viewers_group_member_rclone_read_only_service_account" {
  source    = "../../../../modules/nebius/iam_group_membership"
  parent_id = data.nebius_iam_v1_group.viewers_group.id
  member_id = module.rclone_read_only_service_account.id
  labels    = var.common_labels
}
module "rclone_read_only_service_account" {
  source               = "../../../../modules/nebius/iam_service_account"
  project_id           = var.project_id
  service_account_name = "rclone-read-only-service-account"
  labels               = var.common_labels
}

# Rclone read-write service account
module "editors_group_member_rclone_read_write_service_account" {
  source    = "../../../../modules/nebius/iam_group_membership"
  parent_id = data.nebius_iam_v1_group.editors_group.id
  member_id = module.rclone_read_write_service_account.id
  labels    = var.common_labels
}
module "rclone_read_write_service_account" {
  source               = "../../../../modules/nebius/iam_service_account"
  project_id           = var.project_id
  service_account_name = "rclone-read-write-service-account"
  labels               = var.common_labels
}
