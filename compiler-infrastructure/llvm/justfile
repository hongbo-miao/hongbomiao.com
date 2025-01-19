build:
	clang++ -g src/main.cpp `llvm-config --cxxflags --ldflags --system-libs --libs core executionengine mcjit interpreter native` -o output/main
	./output/main
