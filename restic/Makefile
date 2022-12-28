setup:
	brew install restic
	sudo apt install --yes restic

# Local
restic-init:
	export RESTIC_PASSWORD=passw0rd && \
	restic init --repo=/tmp/restic-backup
restic-backup:
	export RESTIC_PASSWORD=passw0rd && \
	restic backup --repo=/tmp/restic-backup ../hm-alpine
restic-list-snapshots:
	export RESTIC_PASSWORD=passw0rd && \
	restic list --repo=/tmp/restic-backup snapshots
restic-restore:
	export RESTIC_PASSWORD=passw0rd && \
	restic restore --repo=/tmp/restic-backup --target=/tmp/restic-restore xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# SFTP
restic-init-sftp:
	export RESTIC_PASSWORD=passw0rd && \
	restic --repo=sftp:parallels@10.211.55.4:/home/parallels/Documents/hm-backup init
restic-backup-sftp:
	export RESTIC_PASSWORD=passw0rd && \
	restic backup --repo=sftp:parallels@10.211.55.4:/home/parallels/Documents/hm-backup ../hm-alpine
restic-list-snapshots-sftp:
	export RESTIC_PASSWORD=passw0rd && \
	restic list --repo=sftp:parallels@10.211.55.4:/home/parallels/Documents/hm-backup snapshots
restic-restore-sftp:
	export RESTIC_PASSWORD=passw0rd && \
	restic restore --repo=sftp:parallels@10.211.55.4:/home/parallels/Documents/hm-backup --target=/tmp/restic-restore xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
