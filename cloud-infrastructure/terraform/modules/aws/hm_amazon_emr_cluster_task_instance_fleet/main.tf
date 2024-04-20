terraform {
  required_providers {
    aws = {
      source = "hashicorp/aws"
    }
  }
}

# https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/emr_instance_fleet
resource "aws_emr_instance_fleet" "hm_amazon_emr_cluster_task_instance_fleet" {
  cluster_id           = var.amazon_emr_cluster_id
  name                 = "Task"
  target_spot_capacity = var.task_instance_target_spot_capacity
  dynamic "instance_type_configs" {
    for_each = var.task_instance_configs
    content {
      weighted_capacity                          = instance_type_configs.value.weighted_capacity
      instance_type                              = instance_type_configs.value.instance_type
      bid_price_as_percentage_of_on_demand_price = 100
    }
  }
  launch_specifications {
    spot_specification {
      allocation_strategy      = "price-capacity-optimized"
      timeout_duration_minutes = 60
      timeout_action           = "TERMINATE_CLUSTER"
    }
  }
  lifecycle {
    ignore_changes = [
      # https://stackoverflow.com/questions/77442744/how-can-i-make-emr-cluster-auto-scaling-to-utilize-on-demand-instances-while-sta
      target_on_demand_capacity,
      target_spot_capacity
    ]
  }
}
