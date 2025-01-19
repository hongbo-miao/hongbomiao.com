# Host (macOS)
parallels-desktop-show-ip:
	prlctl list --full
generate-ssh-key:
	ssh-keygen -t ed25519
copy-ssh-key-to-ubuntu:
	ssh-copy-id -i ~/.ssh/id_ed25519.pub parallels@10.211.55.4
ssh:
	ssh parallels@10.211.55.4

# Ubuntu
show-system-info:
	uname --all
show-ip:
	hostname --all-ip-addresses
show-usb-devices:
	lsusb
show-kernel-ring-buffer-messages:
	sudo dmesg
show-current-user:
	whoami
show-command-location:
	which xxx

# Install
apt-update:
	sudo apt-get update
apt-install:
	sudo apt --yes xxx
snap-install:
	sudo snap install htop

# Open file descriptor
show-open-file-descriptor-soft-limit:
	ulimit -n
show-open-file-descriptor-hard-limit:
	ulimit -n -H
set-open-file-descriptor-soft-limit:
	ulimit -n 102400

# CPU
show-cpu-cores:
	nproc
show-cpu-summary:
	lscpu
show-cpu-details:
	cat /proc/cpuinfo

# Disk
show-block-devices:
	lsblk
show-disk-free:
	df --human-readable
show-disk-usage:
	du --human-readable --max-depth=0 /home

# Network
show-network-live:
	vnstat --live 1 --style 4 --rateunit 0
show-network-hours:
	vnstat --hours
show-network-hoursgraph:
	vnstat --hoursgraph
show-dns:
	dig www.hongbomiao.com
show-dns-record-all:
	dig www.hongbomiao.com any
show-dns-record-nameserver:
	dig www.hongbomiao.com ns
test-ip-connectivity:
	ping www.hongbomiao.com
test-ssl-connectivity:
	openssl s_client -connect www.hongbomiao.com:443

# SSH
ssh-remove-host-key:
	ssh-keygen -R xxx.xxx.xxx.xxx

# File
ls-owner-group:
	ls -l
	ls -l file.txt

# https://github.com/quic-go/quic-go/wiki/UDP-Receive-Buffer-Size#non-bsd
increase-udp-receive-buffer:
	sysctl -w net.core.rmem_max=2500000

# Service
show-services-active:
	systemctl list-units --type=service --state=active
show-services-active-running:
	systemctl list-units --type=service --state=running
show-services-inactive:
	systemctl list-units --type=service --state=inactive
show-service-status:
	systemctl status xxx
show-service-log:
	journalctl --unit=xxx --lines=60 --no-pager
service-start:
	sudo systemctl start xxx
service-stop:
	sudo systemctl stop xxx
service-restart:
	sudo systemctl restart xxx
service-enable:
	sudo systemctl enable xxx
service-disable:
	sudo systemctl disable xxx

# History
history-show:
	history
history-clear:
	history -c

# Device
show-serial-devices:
	ls /dev/tty*

# GPG
gpg-private-key-list:
	gpg --list-secret-keys
gpg-public-key-list:
	gpg --list-keys
gpg-public-key-show:
	gpg --armor --export xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# Debug
btop:
	btop
htop:
	htop
