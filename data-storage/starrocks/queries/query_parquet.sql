-- Query metadata
desc files (
  "aws.s3.region" = "us-west-2",
  "aws.s3.access_key" = "xxx",
  "aws.s3.secret_key" = "xxx",
  "format" = "parquet",
  "path" = "s3://iot-data-bucket/motor.parquet"
);

-- Query data
select * from files (
  "aws.s3.region" = "us-west-2",
  "aws.s3.access_key" = "xxx",
  "aws.s3.secret_key" = "xxx",
  "format" = "parquet",
  "path" = "s3://iot-data-bucket/motor.parquet"
);
