-- Query metadata
desc function s3 (
  "s3.endpoint" = "https://s3.us-west-2.amazonaws.com",
  "s3.region" = "us-west-2",
  "s3.access_key" = "xxx",
  "s3.secret_key" = "xxx",
  "use_path_style" = "true",
  "format" = "parquet",
  "uri" = "s3://iot-data-bucket/motor.parquet"
);

-- Query data
select * from s3 (
  "s3.endpoint" = "https://s3.us-west-2.amazonaws.com",
  "s3.region" = "us-west-2",
  "s3.access_key" = "xxx",
  "s3.secret_key" = "xxx",
  "format" = "parquet",
  "uri" = "s3://iot-data-bucket/motor.parquet"
);
