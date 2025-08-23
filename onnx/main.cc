#include <iostream>
#include <fstream>
#include <tvm/runtime/relax_vm/executable.h>
#include <tvm/runtime/logging.h>
#include <tvm/runtime/memory/memory_manager.h>
#include <tvm/runtime/data_type.h>
#include <algorithm> // Add this at the top with other includes
#include <chrono>

using namespace std;
using namespace std::chrono;
using tvm::runtime::Module;
using tvm::runtime::PackedFunc;
using tvm::runtime::memory::AllocatorType;

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

const char* labels[] = {"plane",
                        "car",
                        "bird",
                        "cat",
                        "deer",
                        "dog",
                        "frog",
                        "horse",
                        "ship",
                        "truck"
                    };

const int shapes[] = {1, 3, 32, 32};

#define LABELS_COUNT (sizeof(labels) / sizeof(labels[0]))

int main(int argc, char** argv)
{
    std::string path = "./libtvm_model.so";

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
    auto input_img = load_bin(argv[1], shapes[0] * shapes[1] * shapes[2] * shapes[3]);

    std::vector<float> C(LABELS_COUNT, 0.0f);

    // Create and initialize the input array
    tvm::runtime::NDArray IMG = tvm::runtime::NDArray::Empty({shapes[0], shapes[1], shapes[2], shapes[3]}, tvm::runtime::DataType::Float(32), device);
    tvm::runtime::NDArray output_C = tvm::runtime::NDArray::Empty({1, LABELS_COUNT}, tvm::runtime::DataType::Float(32), device);

    memcpy(IMG->data, input_img.data(), sizeof(float) * input_img.size());

    // Measure Take Time
    // Get starting timepoint
    auto start = high_resolution_clock::now();
    output_C = main(IMG);
    // Get ending timepoint
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);

    cout << "Time taken by function: "
         << duration.count() << " microseconds" << endl;

    memcpy(C.data(), output_C->data, sizeof(float) * C.size());

    printf("Output C:\n");
    for (int i = 0; i < LABELS_COUNT; ++i) 
    {
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