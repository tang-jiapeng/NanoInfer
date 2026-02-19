#ifndef NANO_INFER_KERNEL_REGISTEY_H
#define NANO_INFER_KERNEL_REGISTEY_H

#include <glog/logging.h>
#include <functional>
#include <memory>
#include <string>
#include <typeindex>
#include <unordered_map>
#include "nanoinfer/base/base.h"
#include "nanoinfer/base/cuda_config.h"
#include "nanoinfer/tensor/tensor.h"

namespace kernel {

class KernelRegistry {
   public:
    static KernelRegistry& instance() {
        static KernelRegistry reg;
        return reg;
    }

    // 注册: registry.add<AddKernelFn>("add", kDeviceCPU, add_kernel_cpu);
    template <typename Fn>
    void add(const std::string& name, base::DeviceType device, Fn fn) {
        auto key = make_key(name, device, typeid(Fn));
        if (table_.find(key) != table_.end()) {
            LOG(WARNING) << "Kernel [" << name << "] for device [" << static_cast<int>(device)
                         << "] is being overwritten.";
        }
        // 两次转换：函数指针 -> size_t -> void*，避免编译器警告
        table_[key] = reinterpret_cast<void*>(reinterpret_cast<size_t>(fn));
    }

    // 获取: auto k = registry.get<AddKernelFn>("add", kDeviceCPU);
    template <typename Fn>
    Fn get(const std::string& name, base::DeviceType device) const {
        auto key = make_key(name, device, typeid(Fn));
        auto it = table_.find(key);
        CHECK(it != table_.end()) << "Kernel not found: " << name
                                  << " device=" << static_cast<int>(device);
        // 两次转换：void* -> size_t -> 函数指针
        return reinterpret_cast<Fn>(reinterpret_cast<size_t>(it->second));
    }

   private:
    struct Key {
        std::string name;
        base::DeviceType device;
        std::type_index type;

        bool operator==(const Key& o) const {
            return name == o.name && device == o.device && type == o.type;
        }
    };

    struct KeyHash {
        size_t operator()(const Key& k) const {
            size_t h = std::hash<std::string>{}(k.name);
            h ^= std::hash<int>{}(static_cast<int>(k.device)) << 1;
            h ^= k.type.hash_code() << 2;
            return h;
        }
    };

    Key make_key(const std::string& n, base::DeviceType d, std::type_index t) const {
        return Key{n, d, t};
    }

    std::unordered_map<Key, void*, KeyHash> table_;
};

// 自动注册基础设施
template <typename Fn>
struct KernelRegisterer {
    KernelRegisterer(const std::string& name, base::DeviceType device, Fn fn) {
        KernelRegistry::instance().add<Fn>(name, device, fn);
    }
};

// &func 显式获取函数指针，decltype(&func) 确保类型严格匹配
#define REGISTER_KERNEL(name, device_tag, func)                                   \
    static kernel::KernelRegisterer<decltype(&func)> __reg_##name##_##device_tag( \
        #name, base::DeviceType::device_tag, &func);

}  // namespace kernel

#endif  // NANO_INFER_kernel_registry_H