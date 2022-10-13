TVM_ROOT=$(shell cd ../..; pwd)
DMLC_CORE=${TVM_ROOT}/3rdparty/dmlc-core

PKG_CFLAGS = -std=c++17 -g -ggdb -fPIC\
	-I${TVM_ROOT}/include\
	-I${DMLC_CORE}/include\
	-I${TVM_ROOT}/3rdparty/dlpack/include\
	-DDMLC_USE_LOGGING_LIBRARY=\<tvm/runtime/logging.h\>

PKG_LDFLAGS = -L${TVM_ROOT}/build -ldl -pthread

.PHONY: clean all

all: run_cpp

run_cpp: run_cpp_forward.cpp
	$(CXX) $(PKG_CFLAGS) -o $@  $^ -ltvm_runtime $(PKG_LDFLAGS)

clean:
	rm run_cpp