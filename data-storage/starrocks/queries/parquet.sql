-- Query metadata
desc files (
  "aws.s3.region" = "us-west-2",
  "aws.s3.use_aws_sdk_default_behavior" = "true",
  "format" = "parquet",
  "path" = "s3://iot-data-bucket/motor.parquet"
);

-- Query data
select * from files (
  "aws.s3.region" = "us-west-2",
  "aws.s3.use_aws_sdk_default_behavior" = "true",
  "format" = "parquet",
  "path" = "s3://iot-data-bucket/motor.parquet"
);

-- Load data
create database if not exists production_hm_iot_db;
use production_hm_iot_db;

create table motor as
select * from files
(
  "path" = "s3://iot-data-bucket/motor.parquet",
  "format" = "parquet",
  "aws.s3.region" = "us-west-2",
  "aws.s3.use_aws_sdk_default_behavior" = "true"
);
