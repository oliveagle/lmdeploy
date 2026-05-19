// Debug test for LinearWeight::param() - check if it's the right vtable entry
#include <iostream>
#include <cassert>

#include "src/turbomind/core/module.h"
#include "src/turbomind/models/linear_weight.h"

int main()
{
    std::cout << "=== Debugging LinearWeight::param() ===" << std::endl;

    // Create a LinearConfig
    turbomind::core::LinearConfig cfg;
    cfg.input_dim = 128;
    cfg.output_dim = 256;
    cfg.data_type = turbomind::DataType::kFloat16;

    auto module = turbomind::core::Module::create(cfg);
    if (!module) {
        std::cerr << "FAIL: Module::create returned nullptr" << std::endl;
        return 1;
    }

    std::cout << "Module type: " << module->type() << std::endl;

    // Cast to LinearWeight
    auto* linear = dynamic_cast<turbomind::LinearWeight*>(module.get());
    if (!linear) {
        std::cerr << "FAIL: dynamic_cast<LinearWeight*> failed" << std::endl;
        return 1;
    }

    std::cout << "Dynamic cast to LinearWeight: SUCCESS" << std::endl;

    // Check if param() finds "weight" using for_each_param
    std::cout << "\n--- Checking if for_each_param finds 'weight' ---" << std::endl;
    bool found_weight = false;
    linear->for_each_param([&](const char* name, turbomind::core::Tensor&) {
        if (std::string(name) == "weight") {
            found_weight = true;
        }
    });
    std::cout << "  for_each_param found 'weight': " << (found_weight ? "YES" : "NO") << std::endl;

    // Try param("weight") directly
    std::cout << "\n--- Trying param() on LinearWeight ---" << std::endl;
    auto param = linear->param("weight");
    std::cout << "  linear->param(\"weight\"): valid=" << static_cast<bool>(param) << std::endl;

    // Try param("weight") via base class
    auto param2 = module->param("weight");
    std::cout << "  module->param(\"weight\"): valid=" << static_cast<bool>(param2) << std::endl;

    // Check if they're calling the same thing
    std::cout << "\n--- Checking vtable ---" << std::endl;
    // The address of the virtual function
    typedef turbomind::core::Param (turbomind::core::Module::*ParamFn)(const std::string&);
    union {
        ParamFn fn;
        void* ptr;
    } u;
    u.fn = &turbomind::core::Module::param;
    std::cout << "  Module::param vtable entry: " << u.ptr << std::endl;

    // Check if LinearWeight::param is defined
    std::cout << "  LinearWeight::param is defined at: 0x" << std::hex <<
        (void*)(&turbomind::LinearWeight::param) << std::dec << std::endl;

    std::cout << "\n=== Test complete ===" << std::endl;
    return found_weight ? 0 : 1;
}
