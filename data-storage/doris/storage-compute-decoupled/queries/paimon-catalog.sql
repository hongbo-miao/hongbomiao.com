-- Same warehouse the Flink seed job writes to (kubernetes/paimon-seed/seed-sql-configmap.yaml),
-- read here through Doris's own external catalog rather than copied into Doris storage.
create catalog if not exists paimon_catalog properties (
    "type" = "paimon",
    "warehouse" = "s3://paimon-warehouse/",
    "s3.endpoint" = "http://rustfs-svc.rustfs:9000",
    "s3.region" = "us-west-2",
    "s3.access_key" = "rustfs_admin",
    "s3.secret_key" = "passw0rd",
    "use_path_style" = "true"
);

show catalogs;
