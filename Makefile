srcDir := $(shell pwd)
buildDir := $(shell pwd)/build

all: configure build

configure:
	mkdir -p $(buildDir)
	cmake -B $(buildDir) -S $(srcDir) -G Ninja \
	 -DCMAKE_BUILD_TYPE=RelWithDebInfo \
	 -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
	 -DCUDAToolkit_ROOT=/usr/local/cuda-12.3 \
	 -DCMAKE_CUDA_ARCHITECTURES=89 \
	 -DCMAKE_EXPORT_COMPILE_COMMANDS=1

reconfigure:
	rm -rf $(buildDir)
	mkdir -p $(buildDir)
	cmake -B $(buildDir) -S $(srcDir) -G Ninja \
	 -DCMAKE_BUILD_TYPE=RelWithDebInfo \
	 -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
	 -DCUDAToolkit_ROOT=/usr/local/cuda-12.3 \
	 -DCMAKE_CUDA_ARCHITECTURES=89 \
	 -DCMAKE_EXPORT_COMPILE_COMMANDS=1

build:
	cd build && ninja all

test:
	cd build && ctest

.PHONY: build configure reconfigure test


