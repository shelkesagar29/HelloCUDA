srcDir := $(shell pwd)
buildDir := $(shell pwd)/build

all: configure build

configure:
	mkdir -p $(buildDir)
	cmake -B $(buildDir) -S $(srcDir) -G Ninja \
	 -DCMAKE_BUILD_TYPE=RelWithDebInfo \
	 -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++

reconfigure:
	rm -rf $(buildDir)
	mkdir -p $(buildDir)
	cmake -B $(buildDir) -S $(srcDir) -G Ninja \
	 -DCMAKE_BUILD_TYPE=RelWithDebInfo \
	 -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++

build:
	cd build && ninja all

.PHONY: build configure reconfigure


