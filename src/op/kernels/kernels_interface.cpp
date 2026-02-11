#include "kernels_interface.h"
#include "cpu/add_kernel.h"
#include "cpu/embedding_kernel.h"
#include "cpu/matmul_kernel.h"
#include "cpu/mha_kernel.h"
#include "cpu/rmsnorm_kernel.h"
#include "cpu/rope_kernel.h"
#include "cpu/scale_kernel.h"
#include "cpu/scale_sum_kernel.h"
#include "cpu/softmax_kernel.h"
#include "cpu/swiglu_kernel.h"
#include "cuda/add_kernel.cuh"
// #include "cuda/emb_kernel.cuh"
// #include "cuda/matmul_kernel.cuh"
// #include "cuda/mha_kernel.cuh"
// #include "cuda/rmsnorm_kernel.cuh"
// #include "cuda/rope_kernel.cuh"
// #include "cuda/swiglu_kernel.cuh"

namespace kernel {
AddKernel get_add_kernel(base::DeviceType device_type) {
    if (device_type == base::DeviceType::kDeviceCPU) {
        return add_kernel_cpu;
    } else if (device_type == base::DeviceType::kDeviceCUDA) {
        return add_kernel_cu;
    } else {
        LOG(FATAL) << "Unknown device type for get a add kernel.";
        return nullptr;
    }
}

EmbeddingKernel get_embedding_kernel(base::DeviceType device_type) {
    if (device_type == base::DeviceType::kDeviceCPU) {
        return embedding_kernel_cpu;
    } else {
        LOG(FATAL) << "Unknown device type for get a embedding kernel.";
        return nullptr;
    }
}

MatmulKernel get_matmul_kernel(base::DeviceType device_type) {
    if (device_type == base::DeviceType::kDeviceCPU) {
        return matmul_kernel_cpu;
    } else {
        LOG(FATAL) << "Unknown device type for get a matmul kernel.";
        return nullptr;
    }
}

MatmulKernelQuant get_matmul_kernel_quant8(base::DeviceType device_type) {
    if (device_type == base::DeviceType::kDeviceCUDA) {
        return nullptr;
    } else {
        LOG(FATAL) << "Unknown device type for get an matmul kernel.";
        return nullptr;
    }
}

MHAKernel get_mha_kernel(base::DeviceType device_type) {
    if (device_type == base::DeviceType::kDeviceCPU) {
        return mha_kernel_cpu;
    } else {
        LOG(FATAL) << "Unknown device type for get a mha kernel.";
        return nullptr;
    }
}

RoPEKernel get_rope_kernel(base::DeviceType device_type) {
    if (device_type == base::DeviceType::kDeviceCPU) {
        return rope_kernel_cpu;
    } else {
        LOG(FATAL) << "Unknown device type for get a rope kernel.";
        return nullptr;
    }
}

ScaleKernel get_scale_kernel(base::DeviceType device_type) {
    if (device_type == base::DeviceType::kDeviceCPU) {
        return scale_kernel_cpu;
    } else {
        LOG(FATAL) << "Unknown device type for get a scale kernel.";
        return nullptr;
    }
}

SoftmaxKernel get_softmax_kernel(base::DeviceType device_type) {
    if (device_type == base::DeviceType::kDeviceCPU) {
        return softmax_kernel_cpu;
    } else {
        LOG(FATAL) << "Unknown device type for get a softmax kernel.";
        return nullptr;
    }
}

SwigluKernel get_swiglu_kernel(base::DeviceType device_type) {
    if (device_type == base::DeviceType::kDeviceCPU) {
        return swiglu_kernel_cpu;
    } else {
        LOG(FATAL) << "Unknown device type for get a swiglu kernel.";
        return nullptr;
    }
}

RMSNormKernel get_rmsnorm_kernel(base::DeviceType device_type) {
    if (device_type == base::DeviceType::kDeviceCPU) {
        return rmsnorm_kernel_cpu;
    } else {
        LOG(FATAL) << "Unknown device type for get a rmsnorm kernel.";
        return nullptr;
    }
}

ScaleSumKernel get_scale_sum_kernel(base::DeviceType device_type) {
    if (device_type == base::DeviceType::kDeviceCPU) {
        return scale_sum_kernel_cpu;
    } else {
        LOG(FATAL) << "Unknown device type for get a scale sum kernel.";
        return nullptr;
    }
}

}  // namespace kernel