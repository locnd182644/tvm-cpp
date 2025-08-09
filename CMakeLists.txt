cmake_minimum_required(VERSION 3.13)

project(tvm_relax_cpp_example LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(main main.cc)

target_include_directories(main PUBLIC
  /root/Programming/tvm/include
  /root/Programming/tvm/3rdparty/dlpack/include
  /root/Programming/tvm/3rdparty/dmlc-core/include
)

target_link_libraries(main PRIVATE
  /root/Programming/tvm/cross_build/libtvm_runtime.so
)
