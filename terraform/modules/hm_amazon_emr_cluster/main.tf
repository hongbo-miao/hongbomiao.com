# https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/emr_cluster
resource "aws_emr_cluster" "hm_amazon_emr_cluster" {
  name                              = var.amazon_emr_cluster_name
  release_label                     = var.amazon_emr_version
  applications                      = var.applications
  termination_protection            = true
  keep_job_flow_alive_when_no_steps = true
  log_uri                           = "s3://hongbomiao-bucket/amazon-emr/logs/"
  ec2_attributes {
    instance_profile                  = "arn:aws:iam::272394222652:instance-profile/hm-emr-profile"
    subnet_id                         = "subnet-xxxxxxxxxxxxxxxxx"
    emr_managed_master_security_group = "sg-xxxxxxxxxxxxxxxxx"
    emr_managed_slave_security_group  = "sg-xxxxxxxxxxxxxxxxx"
    service_access_security_group     = "sg-xxxxxxxxxxxxxxxxx"
    key_name                          = "hm-ec2-key-pair"
  }
  master_instance_fleet {
    name                      = "Primary"
    target_on_demand_capacity = 1
    launch_specifications {
      on_demand_specification {
        allocation_strategy = "lowest-price"
      }
    }
    instance_type_configs {
      instance_type     = var.primary_instance_type
      weighted_capacity = 1
    }
  }
  core_instance_fleet {
    name                      = "Core"
    target_on_demand_capacity = var.core_target_on_demand_capacity
    launch_specifications {
      on_demand_specification {
        allocation_strategy = "lowest-price"
      }
    }
    instance_type_configs {
      instance_type     = var.core_instance_type
      weighted_capacity = 1
    }
  }
  bootstrap_action {
    name = "set_up"
    path = var.bootstrap_set_up_script_s3_uri
  }
  configurations_json = var.configurations_json_string
  step                = var.steps
  service_role        = var.iam_role_arn
  tags = {
    for-use-with-amazon-emr-managed-policies = true
    Environment                              = var.environment
    Team                                     = var.team
    Name                                     = var.amazon_emr_cluster_name
  }
  # Avoid forcing replacement
  # https://github.com/hashicorp/terraform-provider-aws/issues/12683#issuecomment-752899019
  lifecycle {
    ignore_changes = [
      configurations_json,
      step
    ]
  }
}
