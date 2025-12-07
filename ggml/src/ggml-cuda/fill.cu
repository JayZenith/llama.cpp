#include "fill.cuh"

#define CUDA_FILL_BLOCK_SIZE 256

template <typename T>
static __global__ void fill_kernel(T * __restrict__ dst, const int64_t k, const float value) {
    const int64_t i = (int64_t)blockDim.x * blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }

    dst[i] = (T)value;
}

template <typename T>
static void fill_cuda(T * dst, const int64_t k, const float value, cudaStream_t stream) {
    const int64_t num_blocks = (k + CUDA_FILL_BLOCK_SIZE - 1) / CUDA_FILL_BLOCK_SIZE;
    fill_kernel<T><<<num_blocks, CUDA_FILL_BLOCK_SIZE, 0, stream>>>(dst, k, value);
}

void ggml_cuda_op_fill(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    void * dst_d = dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(dst->type == GGML_TYPE_F32 || dst->type == GGML_TYPE_F16);

    // Extract fill value from op_params (first float)
    float value;
    memcpy(&value, dst->op_params, sizeof(float));

    const int64_t k = ggml_nelements(dst);

    if (dst->type == GGML_TYPE_F16) {
        fill_cuda((half *)dst_d, k, value, stream);
    } else {
        fill_cuda((float *)dst_d, k, value, stream);
    }
}