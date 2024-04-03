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
  connector_configuration = {
    "connector.class" = "com.snowflake.kafka.connector.SnowflakeSinkConnector"
    "tasks.max"       = "8"
    "topics" : "tracker.analytic-events.avro",
    "buffer.count.records" : "10000",
    "buffer.flush.time" : "60",
    "buffer.size.bytes" : "5000000",
    "snowflake.url.name" : "hongbomiao.snowflakecomputing.com:443",
    "snowflake.user.name" : var.snowflake_username,
    "snowflake.private.key" : var.snowflake_private_key,
    "snowflake.private.key.passphrase" : var.snowflake_private_key_passphrase,
    "snowflake.database.name" : "production_tracker_db",
    "snowflake.schema.name" : "public",
    "snowflake.topic2table.map" : "tracker.analytic-events.avro:analytic_events",
    "key.converter" : "org.apache.kafka.connect.storage.StringConverter",
    "value.converter" : "com.snowflake.kafka.connector.records.SnowflakeAvroConverter",
    "value.converter.schema.registry.url" : "https://confluent-schema-registry.hongbomiao.com",
  }
  kafka_cluster {
    apache_kafka_cluster {
      bootstrap_servers = "b-1.hmkafka.xxxxxx.xx.kafka.us-west-2.amazonaws.com:9098,b-2.hmkafka.xxxxxx.xx.kafka.us-west-2.amazonaws.com:9098"
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
