terraform {
  required_providers {
    aws = {
      source = "hashicorp/aws"
    }
  }
}

# https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/mskconnect_connector
resource "aws_mskconnect_connector" "msk_s3_parquet_sink_connector" {
  name = var.amazon_msk_connector_name
  # https://docs.confluent.io/kafka-connectors/s3-sink/current/configuration_options.html
  connector_configuration = {
    "connector.class"   = "io.confluent.connect.s3.S3SinkConnector"
    "topics"            = join(",", var.kafka_topics)
    "tasks.max"         = var.max_task_number
    "errors.log.enable" = true
    "errors.tolerance"  = "all"
    "jmx"               = true

    # Connector
    "flush.size"                  = "1000000"
    "rotate.schedule.interval.ms" = "3600000" # every hour

    # Schema
    "schema.compatibility" = "FULL"

    # S3
    "s3.region"      = var.aws_region
    "s3.bucket.name" = var.s3_bucket_name
    "s3.part.size"   = "104857600" # 100 MiB

    # Storage
    "storage.class" = "io.confluent.connect.s3.storage.S3Storage"

    # Partition
    "partitioner.class"     = "io.confluent.connect.storage.partitioner.TimeBasedPartitioner"
    "path.format"           = "'year'=YYYY/'month'=MM/'day'=dd/'hour'=HH"
    "partition.duration.ms" = 3600000 # 1 hour
    "locale"                = "en"
    "timezone"              = "UTC"

    # Parquet
    "format.class"                        = "io.confluent.connect.s3.format.parquet.ParquetFormat"
    "parquet.codec"                       = "lz4"
    "value.converter"                     = "io.confluent.connect.avro.AvroConverter"
    "value.converter.schema.registry.url" = var.confluent_schema_registry_url
  }
  kafkaconnect_version = var.kafka_connect_version
  capacity {
    # https://docs.aws.amazon.com/MSKC/latest/mskc/API_AutoScaling.html
    autoscaling {
      min_worker_count = 1
      max_worker_count = var.max_worker_number
      mcu_count        = var.worker_microcontroller_unit_number
      scale_in_policy {
        cpu_utilization_percentage = 20
      }
      scale_out_policy {
        cpu_utilization_percentage = 80
      }
    }
  }
  kafka_cluster {
    apache_kafka_cluster {
      bootstrap_servers = var.amazon_msk_cluster_bootstrap_servers
      vpc {
        security_groups = [var.amazon_vpc_security_group_id]
        subnets         = var.amazon_vpc_subnet_ids
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
  tags = merge(var.common_tags, {
    "hm:resource_name" = var.amazon_msk_connector_name
  })
}
