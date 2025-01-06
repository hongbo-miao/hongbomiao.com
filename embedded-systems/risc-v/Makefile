install:
	sudo apt-get install --yes qemu-system-riscv64
	sudo apt-get install --yes qemu-user

run:
	# Assemble the source file
	riscv64-linux-gnu-as -o output/main.o src/main.s
	# Link the object file
	riscv64-linux-gnu-ld -T src/link.ld -o output/main output/main.o
	# Run the executable using QEMU
	qemu-riscv64 output/main
	# Check the result (will print the return value)
	echo $?
