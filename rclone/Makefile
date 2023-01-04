# Config
rclone-config:
	rclone config
rclone-config-create-sftp:
	rclone config create hm-ubuntu sftp user=parallels host=10.211.55.4 port=22 key_file=~/.ssh/id_rsa
rclone-config-create-s3:
	rclone config create hm-s3 s3 provider=AWS access_key_id=xxx secret_access_key=xxx region=us-west-2 storage_class=STANDARD

# List
rclone-list-files-no-modification-time:
	rclone ls hm-ubuntu:/tmp/rclone-backup

rclone-list-files:
	rclone lsl hm-ubuntu:/tmp/rclone-backup
rclone-list-files-s3:
	rclone lsl hm-s3:hongbomiao-bucket/rclone-backup
rclone-list-files-min-age:
	rclone lsl hm-ubuntu:/tmp/rclone-backup --min-age=1w
rclone-list-files-exclude:
	rclone lsl hm-ubuntu:/tmp/rclone-backup --exclude="*.log"
rclone-list-files-include:
	rclone lsl hm-ubuntu:/tmp/rclone-backup --include="*.log"
rclone-list-files-filter:
	rclone lsl hm-ubuntu:/tmp/rclone-backup --filter="+ statistics.log" --filter="- *.log"

rclone-list-directories:
	rclone lsd hm-ubuntu:/tmp/rclone-backup

# Copy
rclone-copy:
	rclone copy --progress /tmp/rclone-backup hm-ubuntu:/tmp/rclone-backup
	rclone copy --progress /tmp/rclone-backup hm-s3:hongbomiao-bucket/rclone-backup
	rclone copy --progress hm-ubuntu:/tmp/rclone-backup /tmp/rclone-backup
	rclone copy --progress hm-ubuntu:/tmp/rclone-backup hm-ubuntu:/tmp/rclone-backup-big
	rclone copy --progress hm-ubuntu:/tmp/rclone-backup hm-s3:hongbomiao-bucket/rclone-backup
rclone-sync:
	rclone sync --progress hm-ubuntu:/tmp/rclone-backup /tmp/rclone-backup

# UI
rclone-ui:
	rclone rcd --rc-web-gui
