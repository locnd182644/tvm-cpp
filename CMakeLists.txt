cmake_minimum_required(VERSION 3.13)

project(tvm-cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(TVM_PATH /home/duyloc/Programming/tvm)

add_executable(main main.cc)

target_include_directories(main PUBLIC
  ${TVM_PATH}/include
  ${TVM_PATH}/3rdparty/dlpack/include
  ${TVM_PATH}/3rdparty/dmlc-core/include
)

target_link_libraries(main PRIVATE
  ${TVM_PATH}/cross_build/libtvm_runtime.so
)

add_custom_target(package ALL
    DEPENDS main
    COMMAND ${CMAKE_COMMAND} -E tar "cf" release.tar ${CMAKE_BINARY_DIR}/main ${CMAKE_BINARY_DIR}/../weights ${CMAKE_BINARY_DIR}/../*.so
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "TVM library packaging"
)