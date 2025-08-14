## Building the Example Project

Once TVM is built, follow these steps to build the example project:

1. **Configure the build using CMake**:
   ```bash
   cmake -DCMAKE_TOOLCHAIN_FILE=/path/to/toolchain.cmake -DCMAKE_CXX_FLAGS="-DDMLC_LOG_STACK_TRACE=0" -DCMAKE_BUILD_TYPE=Release -B build -G Ninja
   ```

2. **Build the project using Ninja**:
   ```bash
   ninja -C build
   ```