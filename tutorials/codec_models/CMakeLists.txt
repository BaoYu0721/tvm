cmake_minimum_required(VERSION 3.12.0)
project( TVMDeployAndCmp )

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fexceptions -std=c++17 -fPIC")
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} -g -ggdb")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} -g -ggdb")
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} -O3")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} -O3")

set(TVM_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../..)
set(TVM_INCLUDE_DIR 
    ${TVM_ROOT}/include
    ${TVM_ROOT}/3rdparty/dmlc-core/include
    ${TVM_ROOT}/3rdparty/dlpack/include
)
set(TVM_LIB_DIR ${TVM_ROOT}/build)

find_library(TVM_LIB_PATH tvm_runtime HINTS ${TVM_LIB_DIR})

set(TVM_EXE "cpp_deploy_tvm")
add_executable(${TVM_EXE} 
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp_deploy_tvm/run_cpp_forward.cpp
)
include_directories(${TVM_INCLUDE_DIR})
target_link_libraries(${TVM_EXE} PUBLIC
    -ltvm_runtime
)