# Build and run step by step (LLVM IR -> bitcode -> assembly -> executable)
build-step-by-step:
	# 1. Convert LLVM IR to bitcode
	llvm-as src/main.ll -o=output/main.bc
	# 2. Compile bitcode to assembly
	llc output/main.bc -o=output/main.s
	# 3. Compile assembly to executable
	clang output/main.s -o=output/main
	# 4. Run the program
	./output/main
	# 5. Check the result (will print the return value)
	echo $?

# Build and run in one step (LLVM IR -> executable)
build-single-step:
	clang src/main.ll -o output/main
	./output/main
	echo $?
