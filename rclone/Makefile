setup:
	brew install rclone

rclone-config:
	rclone config
rclone-config-create:
	rclone config create hm-ubuntu sftp user=parallels host=10.211.55.4 port=22 key_file=~/.ssh/id_rsa

# List files
rclone-list-files:
	rclone ls hm-ubuntu:/tmp/rclone-backup
rclone-list-files-with-modification-time:
	rclone lsl hm-ubuntu:/tmp/rclone-backup
rclone-list-directories-with-modification-time:
	rclone lsd hm-ubuntu:/tmp/rclone-backup

rclone-copy:
	rclone copy --progress /tmp/rclone-backup hm-ubuntu:/tmp/rclone-backup
	rclone copy --progress hm-ubuntu:/tmp/rclone-backup /tmp/rclone-backup
rclone-sync:
	rclone sync --progress /tmp/rclone-backup hm-ubuntu:/tmp/rclone-backup
	rclone sync --progress hm-ubuntu:/tmp/rclone-backup /tmp/rclone-backup

rclone-ui:
	rclone rcd --rc-web-gui
