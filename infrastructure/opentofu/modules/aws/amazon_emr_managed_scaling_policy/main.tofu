terraform {
  required_providers {
    aws = {
      source = "registry.opentofu.org/hashicorp/aws"
    }
  }
}

# https://search.opentofu.org/provider/hashicorp/aws/latest/docs/resources/emr_managed_scaling_policy
resource "aws_emr_managed_scaling_policy" "main" {
  cluster_id = var.amazon_emr_cluster_id
  compute_limits {
    unit_type                       = "InstanceFleetUnits"
    minimum_capacity_units          = 1
    maximum_capacity_units          = var.max_capacity_units
    maximum_ondemand_capacity_units = 1
    maximum_core_capacity_units     = 1
  }
}
