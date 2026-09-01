-- The DorisDisaggregatedCluster CRD has no S3/vault field, so internal-table storage is
-- configured here, after FE is up, rather than at deploy time.
create storage vault if not exists doris_storage_vault
properties (
    "type" = "S3",
    "s3.endpoint" = "rustfs-svc.rustfs:9000",
    "s3.region" = "us-west-2",
    "s3.bucket" = "doris-storage-vault",
    "s3.root.path" = "doris",
    "s3.access_key" = "rustfs_admin",
    "s3.secret_key" = "passw0rd",
    "provider" = "S3",
    "use_path_style" = "true"
);

set doris_storage_vault as default storage vault;

show storage vaults;
