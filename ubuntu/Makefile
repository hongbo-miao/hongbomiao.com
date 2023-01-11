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
show-disk-space:
	df --human-readable
show-cpu:
	cat /proc/cpuinfo
show-ip:
	hostname --all-ip-addresses
show-usb-devices:
	lsusb

apt-update:
	sudo apt update
apt-install:
	sudo apt --yes xxx
