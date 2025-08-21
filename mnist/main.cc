#include <iostream>
#include <fstream>
#include <tvm/runtime/relax_vm/executable.h>
#include <tvm/runtime/logging.h>
#include <tvm/runtime/memory/memory_manager.h>
#include <tvm/runtime/data_type.h>
#include <algorithm> // Add this at the top with other includes

using tvm::runtime::Module;
using tvm::runtime::PackedFunc;
using tvm::runtime::memory::AllocatorType;
using tvm::runtime::relax_vm::VMExecutable;

std::vector<float> load_bin(const char* path, size_t expected_elems) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error(std::string("Cannot open ") + path);
    }
    in.seekg(0, std::ios::end);
    size_t bytes = in.tellg();
    in.seekg(0, std::ios::beg);
    if (bytes != expected_elems * sizeof(float)) {
        std::cerr << "Warning: expected " << expected_elems*sizeof(float) << " bytes, got " << bytes << " bytes\n";
    }
    std::vector<float> v(expected_elems);
    in.read(reinterpret_cast<char*>(v.data()), std::min(bytes, expected_elems*sizeof(float)));
    return v;
}

const char* labels[] = {"T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"};  

int main()
{
  std::string path = "./linear_relu_mnist.so";

  // Load the shared object
  Module m = Module::LoadFromFile(path);
  std::cout << m << std::endl;

  PackedFunc vm_load_executable = m.GetFunction("vm_load_executable");
  CHECK(vm_load_executable != nullptr)
      << "Error: `vm_load_executable` does not exist in file `" << path << "`";

  // Create a VM from the Executable
  Module mod = vm_load_executable();
  PackedFunc vm_initialization = mod.GetFunction("vm_initialization");
  CHECK(vm_initialization != nullptr)
      << "Error: `vm_initialization` does not exist in file `" << path << "`";

  // Initialize the VM
  tvm::Device device{kDLCPU, 0};
  vm_initialization(static_cast<int>(device.device_type), static_cast<int>(device.device_id),
                    static_cast<int>(AllocatorType::kPooled), static_cast<int>(kDLCPU), 0,
                    static_cast<int>(AllocatorType::kPooled));

  PackedFunc main = mod.GetFunction("main");
  CHECK(main != nullptr)
      << "Error: Entry function does not exist in file `" << path << "`";

  // Load binary data
  auto w0 = load_bin("weights/w0.bin", 128 * 784);
  auto b0 = load_bin("weights/b0.bin", 128);
  auto w1 = load_bin("weights/w1.bin", 128 * 10);
  auto b1 = load_bin("weights/b1.bin", 10);
  auto input_img = load_bin("weights/input_img.bin", 784);
  
  std::vector<float> C(10, 0.0f);
  
  // Create and initialize the input array
  tvm::runtime::NDArray IMG = tvm::runtime::NDArray::Empty({1, 784}, tvm::runtime::DataType::Float(32), device);
  tvm::runtime::NDArray W0 = tvm::runtime::NDArray::Empty({128, 784}, tvm::runtime::DataType::Float(32), device);
  tvm::runtime::NDArray B0 = tvm::runtime::NDArray::Empty({128}, tvm::runtime::DataType::Float(32), device);
  tvm::runtime::NDArray W1 = tvm::runtime::NDArray::Empty({10, 128}, tvm::runtime::DataType::Float(32), device);
  tvm::runtime::NDArray B1 = tvm::runtime::NDArray::Empty({10}, tvm::runtime::DataType::Float(32), device);
  tvm::runtime::NDArray output_C = tvm::runtime::NDArray::Empty({1, 10}, tvm::runtime::DataType::Float(32), device);

  memcpy(IMG->data, input_img.data(), sizeof(float) * input_img.size());
  memcpy(W0->data, w0.data(), sizeof(float) * w0.size());
  memcpy(B0->data, b0.data(), sizeof(float) * b0.size());
  memcpy(W1->data, w1.data(), sizeof(float) * w1.size());
  memcpy(B1->data, b1.data(), sizeof(float) * b1.size());

  output_C = main(IMG, W0, B0, W1, B1);

  memcpy(C.data(), output_C->data, sizeof(float) * C.size());
  
  printf("Output C:\n");
  for (int i = 0; i < 10; ++i) {
    std::cout << C[i] << " ";
  }
  std::cout << std::endl;

  // Find max element and its index
  auto max_iter = std::max_element(C.begin(), C.end());
  int max_index = std::distance(C.begin(), max_iter);
  float max_value = *max_iter;
  std::cout << "Max value: " << max_value << " at index: " << max_index << std::endl;
  std::cout << "Predicted label: " << labels[max_index] << std::endl;

  return 0;
}