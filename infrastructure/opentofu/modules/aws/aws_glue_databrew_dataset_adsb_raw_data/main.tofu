terraform {
  required_providers {
    awscc = {
      source = "registry.opentofu.org/hashicorp/awscc"
    }
  }
}

# https://search.opentofu.org/provider/hashicorp/awscc/latest/docs/resources/databrew_dataset
resource "awscc_databrew_dataset" "glue_databrew_dataset_adsb_raw_data" {
  name = var.aws_glue_databrew_dataset_name
  input = {
    s3_input_definition = {
      bucket = var.input_s3_bucket_name
      key    = var.input_s3_key
    }
  }
  format = "JSON"
  format_options = {
    json = {
      multi_line = true
    }
  }
  tags = concat(
    [for key, value in var.common_tags : {
      key   = key
      value = value
    }],
    [
      {
        key   = "hm_resource_name"
        value = var.aws_glue_databrew_dataset_name
      }
    ]
  )
}
