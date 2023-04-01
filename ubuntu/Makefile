# Host (macOS)
generate-ssh-key:
	ssh-keygen -t ed25519
copy-ssh-key-to-ubuntu:
	ssh-copy-id -i ~/.ssh/id_ed25519.pub hongbo-miao@192.168.64.4
	ssh-copy-id -i ~/.ssh/id_ed25519.pub parallels@10.211.55.4
ssh:
	ssh hongbo-miao@192.168.64.4
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

# Open file descriptors
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
show-disk-free:
	df --human-readable
show-disk-usage:
	du --human-readable --max-depth=0 /home

# Network
show-network-live:
	vnstat --live
show-network-hours:
	vnstat --hours
show-network-hoursgraph:
	vnstat --hoursgraph

apt-update:
	sudo apt update
apt-install:
	sudo apt --yes xxx

ls-owner-group:
	ls -l

# https://github.com/quic-go/quic-go/wiki/UDP-Receive-Buffer-Size#non-bsd
increase-udp-receive-buffer:
	sysctl -w net.core.rmem_max=2500000

# Service
show-active-services:
	systemctl list-units --type=service
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

# htop
htop-install:
	sudo snap install htop
htop:
	htop
