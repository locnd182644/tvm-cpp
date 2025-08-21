cmake_minimum_required(VERSION 3.13)

project(tvm-cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(TVM_PATH /root/Programming/tvm)

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
    DEPENDS main data
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_BINARY_DIR}/../*.so ${CMAKE_BINARY_DIR}
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_BINARY_DIR}/data
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/data
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_BINARY_DIR}/../data ${CMAKE_BINARY_DIR}/data
    COMMAND ${CMAKE_COMMAND} -E tar "cf" release.tar ${CMAKE_BINARY_DIR}/main ${CMAKE_BINARY_DIR}/data ${CMAKE_BINARY_DIR}/*.so
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "TVM library packaging"
)