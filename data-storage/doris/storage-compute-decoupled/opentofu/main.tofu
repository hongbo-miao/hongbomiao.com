# root/admin need no policy here -- default.policy.users covers them (see bootstrap-job.yaml).
#
# Row filter and data mask policies grant nothing on their own; the access policy has to exist
# alongside them. It carries both users because Ranger allows only one policy per resource
# signature per service, and both read the same paimon_catalog/*/*/*.
resource "ranger_policy" "paimon_catalog_access" {
  service    = "doris"
  name       = "paimon-catalog-select"
  definition = file("${path.module}/policies/paimon-catalog-access.json")
}

resource "ranger_policy" "application_a_compute_group" {
  service    = "doris"
  name       = "application-a-compute-group-usage"
  definition = file("${path.module}/policies/application-a-compute-group.json")
}

resource "ranger_policy" "application_b_compute_group" {
  service    = "doris"
  name       = "application-b-compute-group-usage"
  definition = file("${path.module}/policies/application-b-compute-group.json")
}

resource "ranger_policy" "application_a_row_filter" {
  service    = "doris"
  name       = "application-a-sensor-reading-row-filter"
  definition = file("${path.module}/policies/application-a-row-filter.json")
}

resource "ranger_policy" "application_a_data_mask" {
  service    = "doris"
  name       = "application-a-sensor-reading-mask-reading"
  definition = file("${path.module}/policies/application-a-data-mask.json")
}
