terraform {
  required_providers {
    aws = {
      source = "hashicorp/aws"
    }
  }
}

# https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/mskconnect_connector
resource "aws_mskconnect_connector" "hm_amazon_msk_connector" {
  name                 = var.amazon_msk_connector_name
  kafkaconnect_version = var.kafka_connect_version
  capacity {
    autoscaling {
      mcu_count        = 1
      min_worker_count = 1
      max_worker_count = 2
      scale_in_policy {
        cpu_utilization_percentage = 40
      }
      scale_out_policy {
        cpu_utilization_percentage = 95
      }
    }
  }
  # https://docs.snowflake.com/en/user-guide/kafka-connector-install#label-kafka-properties
  # https://docs.snowflake.com/en/user-guide/data-load-snowpipe-streaming-kafka
  connector_configuration = {
    "connector.class"                  = "com.snowflake.kafka.connector.SnowflakeSinkConnector"
    "tasks.max"                        = 4
    "topics"                           = var.kafka_topic_name,
    "buffer.count.records"             = 10000,
    "buffer.flush.time"                = 5,
    "buffer.size.bytes"                = 20000000,
    "snowflake.url.name"               = "hongbomiao.snowflakecomputing.com:443",
    "snowflake.user.name"              = var.snowflake_user_name,
    "snowflake.private.key"            = var.snowflake_private_key,
    "snowflake.private.key.passphrase" = var.snowflake_private_key_passphrase,
    "snowflake.role.name"              = var.snowflake_role_name
    "snowflake.ingestion.method"       = "SNOWPIPE_STREAMING"
    "snowflake.enable.schematization"  = true
    "snowflake.database.name"          = var.snowflake_database_name,
    "snowflake.schema.name"            = var.snowflake_schema_name,
    # e.g., development.tracker.analytic-events.avro -> tracker_analytic_events
    "snowflake.topic2table.map" = "${var.kafka_topic_name}:${
      upper(join("_",
        [
          replace(split(".", var.kafka_topic_name)[1], "-", "_"),
          replace(split(".", var.kafka_topic_name)[2], "-", "_")
        ]
      ))
    }",
    "value.converter"                     = "io.confluent.connect.avro.AvroConverter",
    "value.converter.schema.registry.url" = "https://confluent-schema-registry.hongbomiao.com",
    "errors.log.enable"                   = true
    "errors.tolerance"                    = "all"
    "jmx"                                 = true
  }
  kafka_cluster {
    apache_kafka_cluster {
      bootstrap_servers = var.amazon_msk_cluster_bootstrap_servers
      vpc {
        security_groups = [
          "sg-xxxxxxxxxxxxxxxxx"
        ]
        subnets = [
          "subnet-xxxxxxxxxxxxxxxxx",
          "subnet-xxxxxxxxxxxxxxxxx",
          "subnet-xxxxxxxxxxxxxxxxx"
        ]
      }
    }
  }
  kafka_cluster_client_authentication {
    authentication_type = "IAM"
  }
  kafka_cluster_encryption_in_transit {
    encryption_type = "TLS"
  }
  plugin {
    custom_plugin {
      arn      = var.amazon_msk_plugin_arn
      revision = var.amazon_msk_plugin_revision
    }
  }
  log_delivery {
    worker_log_delivery {
      s3 {
        bucket  = var.msk_log_s3_bucket_name
        prefix  = var.msk_log_s3_key
        enabled = true
      }
    }
  }
  service_execution_role_arn = var.amazon_msk_connector_iam_role_arn
}
