terraform {
  required_providers {
    awscc = {
      source = "hashicorp/awscc"
    }
  }
}

# https://registry.terraform.io/providers/hashicorp/awscc/latest/docs/resources/databrew_dataset
resource "awscc_databrew_dataset" "glue_databrew_dataset_adsb_raw_parquet" {
  name = var.aws_glue_databrew_dataset_name
  input = {
    s3_input_definition = {
      bucket = var.input_s3_bucket_name
      key    = var.input_s3_key
    }
  }
  format = "PARQUET"
  tags = concat(
    [for key, value in var.common_tags : {
      key   = key
      value = value
    }],
    [
      {
        key   = "hm:resource_name"
        value = var.aws_glue_databrew_dataset_name
      }
    ]
  )
}
