# https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/emr_cluster
resource "aws_emr_cluster" "hm_amazon_emr_cluster" {
  name                              = var.amazon_emr_cluster_name
  release_label                     = var.amazon_emr_version
  applications                      = ["Trino"]
  termination_protection            = false
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
  master_instance_group {
    name          = "Primary"
    instance_type = var.primary_instance_type
  }
  core_instance_group {
    name           = "Core"
    instance_type  = var.core_instance_type
    instance_count = var.core_instance_count
    bid_price      = 0.3
  }
  bootstrap_action {
    name = "set_up"
    path = "s3://hongbomiao-bucket/amazon-emr/hm-amazon-emr-cluster-trino/bootstrap-actions/set_up.sh"
  }
  configurations_json = jsonencode(
    [
      {
        Classification : "delta-defaults",
        Properties : {
          "delta.enabled" : true
        }
      },
      {
        Classification : "trino-connector-delta",
        Properties : {
          "hive.metastore" : "glue"
        }
      },
      {
        Classification : "trino-connector-postgresql",
        Properties : {
          connection-url : "jdbc:postgresql://${jsondecode(data.aws_secretsmanager_secret_version.hm_rds_secret_version.secret_string)["postgres_host"]}/${jsondecode(data.aws_secretsmanager_secret_version.hm_rds_secret_version.secret_string)["postgres_db"]}",
          connection-user : jsondecode(data.aws_secretsmanager_secret_version.hm_rds_secret_version.secret_string)["postgres_user"],
          connection-password : jsondecode(data.aws_secretsmanager_secret_version.hm_rds_secret_version.secret_string)["postgres_password"]
        }
      }
    ]
  )
  service_role = var.aws_iam_role
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
      configurations_json
    ]
  }
}

data "aws_secretsmanager_secret" "hm_rds_secret" {
  name = "hm-iot-rds/hm_iot_db/readonly"
}
data "aws_secretsmanager_secret_version" "hm_rds_secret_version" {
  secret_id = data.aws_secretsmanager_secret.hm_rds_secret.id
}
