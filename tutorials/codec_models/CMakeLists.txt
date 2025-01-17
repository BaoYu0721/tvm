cmake_minimum_required(VERSION 3.12.0)
project( TVMDeployAndCmp )

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fexceptions -std=c++17 -fPIC")
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} -g -ggdb")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} -g -ggdb")
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} -O3")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} -O3")

set(PROJECT_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)

set(TVM_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../..)
set(TVM_INCLUDE_DIR 
    ${TVM_ROOT}/include
    ${TVM_ROOT}/3rdparty/dmlc-core/include
    ${TVM_ROOT}/3rdparty/dlpack/include
)
set(TVM_LIB_DIR ${TVM_ROOT}/build)

set(yaml-cpp_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/yaml-cpp/include)
set(yaml-cpp_LIB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/yaml-cpp/build/)

find_library(TVM_LIB_PATH tvm_runtime HINTS ${TVM_LIB_DIR})
find_library(YAML_LIB_PATH yaml-cpp HINTS ${yaml-cpp_LIB_DIR})

set(TVM_EXE "cpp_deploy_tvm")
add_executable(${TVM_EXE} 
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp_deploy_tvm/run_cpp_forward.cpp
)
include_directories(${TVM_INCLUDE_DIR} ${PROJECT_INCLUDE_DIR} ${yaml-cpp_INCLUDE_DIR})
target_link_libraries(${TVM_EXE} PUBLIC
    ${TVM_LIB_PATH}
    ${YAML_LIB_PATH}
)

set(TRT_INCLUDE_DIR 
    /home/SENSETIME/baoyu/Downloads/TensorRT-8.4.1.5/include
    /usr/local/cuda/include
)
set(TRT_LIB_DIR 
    /home/SENSETIME/baoyu/Downloads/TensorRT-8.4.1.5/lib
    /usr/local/cuda/lib64
)
set(TRT_EXE "pure_trt_exe")
add_executable(${TRT_EXE} 
    ${CMAKE_CURRENT_SOURCE_DIR}/pure_trt/pure_trt.cpp
)
include_directories(${TRT_INCLUDE_DIR} ${PROJECT_INCLUDE_DIR} ${yaml-cpp_INCLUDE_DIR})
target_link_libraries(${TRT_EXE} PUBLIC
    -lnvinfer
    -lcudart
    -lnvonnxparser
    ${YAML_LIB_PATH}
)
