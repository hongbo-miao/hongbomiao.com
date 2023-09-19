# https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/emr_instance_fleet
resource "aws_emr_instance_fleet" "hm_amazon_emr_cluster_task_instance_fleet" {
  cluster_id           = var.amazon_emr_cluster_id
  name                 = "Task"
  target_spot_capacity = var.task_target_spot_capacity
  instance_type_configs {
    instance_type                              = var.task_instance_type
    weighted_capacity                          = 1
    bid_price_as_percentage_of_on_demand_price = 100
  }
  launch_specifications {
    spot_specification {
      allocation_strategy      = "price-capacity-optimized"
      timeout_duration_minutes = 60
      timeout_action           = "TERMINATE_CLUSTER"
    }
  }
}
