#ifndef NANO_INFER_KERNELS_INTERFACE_H
#define NANO_INFER_KERNELS_INTERFACE_H
#include "nanoinfer/base/cuda_config.h"
#include "nanoinfer/tensor/tensor.h"

namespace kernel {

/**
 * @brief 向量加法 Kernel 函数指针定义
 *
 * 协议规范：
 * 1. 输入输出 Tensor 必须已经分配好内存。
 * 2. stream 参数用于 CUDA 异步执行，CPU 模式下通常忽略。
 * 3. input1, input2, output 的维度必须一致（暂不支持广播）。
 *
 * @param input1 输入张量 1
 * @param input2 输入张量 2
 * @param output 输出张量
 * @param stream CUDA 流句柄 (cudaStream_t)，若为 nullptr 则使用默认流或 CPU 执行
 */
typedef void (*AddKernel)(const tensor::Tensor& input1, const tensor::Tensor& input2,
                          const tensor::Tensor& output, void* stream);

/**
 * @brief RMSNorm Kernel 函数指针定义
 *
 * 协议规范：
 * 1. input 和 output 维度必须一致。
 * 2. weight 的维度必须等于 input 的最后一维。
 *
 * @param input 输入张量
 * @param weight 归一化权重 (gamma)
 * @param output 输出张量
 * @param stream CUDA 流句柄
 */
typedef void (*RMSNormKernel)(const tensor::Tensor& input, const tensor::Tensor& weight,
                              const tensor::Tensor& output, void* stream);

/**
 * @brief 获取对应设备的 Add Kernel 实现
 *
 * @param device_type 设备类型 (kDeviceCPU 或 kDeviceCUDA)
 * @return AddKernel 函数指针。如果设备不支持或未实现，返回 nullptr 或触发 FATAL 错误。
 */
AddKernel get_add_kernel(base::DeviceType device_type);

/**
 * @brief 获取对应设备的 RMSNorm Kernel 实现
 *
 * @param device_type 设备类型
 * @return RMSNormKernel 函数指针
 */
RMSNormKernel get_rmsnorm_kernel(base::DeviceType device_type);

}  // namespace kernel

#endif  // NANO_INFER_KERNELS_INTERFACE_H
