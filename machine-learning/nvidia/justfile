gpu-info:
	nvidia-smi


cuda-toolkit-install-windows:
	# https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64&target_version=11&target_type=exe_local

cuda-toolkit-install-wsl-2:
	# https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=WSL-Ubuntu&target_version=2.0&target_type=runfile_local
	wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
	sudo dpkg --install cuda-keyring_1.1-1_all.deb
	sudo apt-get update
	sudo apt-get --yes install cuda-toolkit-12-6
cuda-toolkit-version:
	nvcc --version
