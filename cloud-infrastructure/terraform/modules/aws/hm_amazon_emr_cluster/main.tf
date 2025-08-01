terraform {
  required_providers {
    aws = {
      source = "hashicorp/aws"
    }
  }
}

# https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/emr_cluster
resource "aws_emr_cluster" "main" {
  name                              = var.amazon_emr_cluster_name
  release_label                     = var.amazon_emr_version
  applications                      = var.applications
  termination_protection            = false
  unhealthy_node_replacement        = true
  keep_job_flow_alive_when_no_steps = true
  log_uri                           = "s3://hm-production-bucket/amazon-emr/logs/"
  ec2_attributes {
    instance_profile                  = "arn:aws:iam::272394222652:instance-profile/hm-emr-profile"
    subnet_id                         = "subnet-xxxxxxxxxxxxxxxxx"
    emr_managed_master_security_group = "sg-xxxxxxxxxxxxxxxxx"
    emr_managed_slave_security_group  = "sg-xxxxxxxxxxxxxxxxx"
    service_access_security_group     = "sg-xxxxxxxxxxxxxxxxx"
    key_name                          = "hm-key-pair"
  }
  master_instance_fleet {
    name                      = "Primary"
    target_on_demand_capacity = var.primary_instance_target_on_demand_capacity
    launch_specifications {
      on_demand_specification {
        allocation_strategy = "lowest-price"
      }
    }
    instance_type_configs {
      weighted_capacity = 1
      instance_type     = var.primary_instance_type
      ebs_config {
        size                 = var.primary_instance_ebs_volume_size_gb
        type                 = "gp3"
        volumes_per_instance = 1
      }
    }
  }
  core_instance_fleet {
    name                      = "Core"
    target_on_demand_capacity = var.core_instance_target_on_demand_capacity
    launch_specifications {
      on_demand_specification {
        allocation_strategy = "lowest-price"
      }
    }
    instance_type_configs {
      weighted_capacity = var.core_instance_weighted_capacity
      instance_type     = var.core_instance_type
      ebs_config {
        size                 = var.core_instance_ebs_volume_size_gb
        type                 = "gp3"
        volumes_per_instance = 1
      }
    }
  }
  bootstrap_action {
    name = "set_up"
    path = var.bootstrap_set_up_script_s3_uri
  }
  configurations_json    = var.configurations_json_string
  placement_group_config = var.placement_group_config
  step                   = var.steps
  service_role           = var.iam_role_arn
  tags = merge(var.common_tags, {
    "hm:resource_name"                         = var.amazon_emr_cluster_name,
    "for-use-with-amazon-emr-managed-policies" = true
  })
  lifecycle {
    create_before_destroy = true
    ignore_changes = [
      # https://github.com/hashicorp/terraform-provider-aws/issues/12683#issuecomment-752899019
      configurations_json,
      # https://github.com/hashicorp/terraform-provider-aws/issues/36492
      core_instance_fleet[0].target_on_demand_capacity,
      step
    ]
  }
}
