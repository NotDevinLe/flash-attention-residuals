#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

namespace {

constexpr int kBlockBt = 4;
constexpr int kBlockX = 128;
constexpr int kMaxQueries = 16;
constexpr int kMaxSourceBlocks = 16;

__device__ __forceinline__ float load_scalar(const float* ptr) {
    return *ptr;
}

__device__ __forceinline__ float load_scalar(const __nv_bfloat16* ptr) {
    return __bfloat162float(*ptr);
}

__device__ __forceinline__ void store_scalar(float* ptr, float value) {
    *ptr = value;
}

__device__ __forceinline__ void store_scalar(__nv_bfloat16* ptr, float value) {
    *ptr = __float2bfloat16(value);
}

template <typename GradOutT, typename GradBlockT>
__global__ void phase_1_backward_kernel(
    const __nv_bfloat16* __restrict__ block_representations,
    const __nv_bfloat16* __restrict__ pseudo_queries,
    const float* __restrict__ lses,
    const float* __restrict__ inverse_rms_norms,
    const float* __restrict__ attention_logits,
    const GradOutT* __restrict__ grad_softmax_outputs,
    const float* __restrict__ grad_lses,
    GradBlockT* __restrict__ grad_block_representations,
    float* __restrict__ grad_pseudo_queries,
    int num_source_blocks,
    int num_batch_seq,
    int hidden_dim,
    int num_queries,
    bool has_grad_lses,
    bool accumulate_grad_blocks) {
    extern __shared__ float shared[];

    const int lane = threadIdx.x;
    const int local_bt = threadIdx.y;
    const int batch_seq_idx = blockIdx.x * kBlockBt + local_bt;
    const bool valid_bt = batch_seq_idx < num_batch_seq;

    float* reduce_scratch = shared;
    float* softmax_probabilities =
        reduce_scratch + kBlockBt * num_source_blocks * kBlockX;
    float* grad_attention_logits =
        softmax_probabilities + num_queries * kBlockBt * num_source_blocks;
    float* pseudo_query_scratch =
        grad_attention_logits + num_queries * kBlockBt * num_source_blocks;

    for (int query_idx = 0; query_idx < num_queries; ++query_idx) {
        for (int source_idx = 0; source_idx < num_source_blocks; ++source_idx) {
            float partial_dot = 0.0f;

            if (valid_bt) {
                const int64_t source_base =
                    (static_cast<int64_t>(source_idx) * num_batch_seq +
                     batch_seq_idx) *
                    hidden_dim;
                const int64_t grad_base =
                    (static_cast<int64_t>(query_idx) * num_batch_seq +
                     batch_seq_idx) *
                    hidden_dim;

                for (int hidden_idx = lane; hidden_idx < hidden_dim;
                     hidden_idx += kBlockX) {
                    const float value = load_scalar(
                        block_representations + source_base + hidden_idx);
                    const float grad_output = load_scalar(
                        grad_softmax_outputs + grad_base + hidden_idx);
                    partial_dot += value * grad_output;
                }
            }

            const int reduce_base =
                (local_bt * num_source_blocks + source_idx) * kBlockX;
            reduce_scratch[reduce_base + lane] = partial_dot;
            __syncthreads();

            for (int stride = kBlockX / 2; stride > 0; stride >>= 1) {
                if (lane < stride) {
                    reduce_scratch[reduce_base + lane] +=
                        reduce_scratch[reduce_base + lane + stride];
                }
                __syncthreads();
            }
        }

        if (lane == 0) {
            float expected_dot = 0.0f;
            const float grad_lse =
                (valid_bt && has_grad_lses)
                    ? grad_lses[static_cast<int64_t>(query_idx) *
                                    num_batch_seq +
                                batch_seq_idx]
                    : 0.0f;
            const float forward_lse =
                valid_bt
                    ? lses[static_cast<int64_t>(query_idx) * num_batch_seq +
                           batch_seq_idx]
                    : 0.0f;

            for (int source_idx = 0; source_idx < num_source_blocks; ++source_idx) {
                float probability = 0.0f;
                if (valid_bt) {
                    const float logit =
                        attention_logits
                            [static_cast<int64_t>(query_idx) * num_batch_seq *
                                 num_source_blocks +
                             static_cast<int64_t>(batch_seq_idx) *
                                 num_source_blocks +
                             source_idx];
                    probability = __expf(logit - forward_lse);
                }

                softmax_probabilities
                    [(query_idx * kBlockBt + local_bt) * num_source_blocks +
                     source_idx] = probability;
                expected_dot +=
                    probability *
                    reduce_scratch
                        [(local_bt * num_source_blocks + source_idx) * kBlockX];
            }

            for (int source_idx = 0; source_idx < num_source_blocks; ++source_idx) {
                const float probability =
                    softmax_probabilities
                        [(query_idx * kBlockBt + local_bt) * num_source_blocks +
                         source_idx];
                const float source_dot =
                    reduce_scratch
                        [(local_bt * num_source_blocks + source_idx) * kBlockX];
                grad_attention_logits
                    [(query_idx * kBlockBt + local_bt) * num_source_blocks +
                     source_idx] =
                        probability * (grad_lse + source_dot - expected_dot);
            }
        }
        __syncthreads();
    }

    for (int hidden_base = 0; hidden_base < hidden_dim; hidden_base += kBlockX) {
        const int hidden_idx = hidden_base + lane;
        const bool valid_hidden = hidden_idx < hidden_dim;

        float pseudo_query_accumulator[kMaxQueries];

        for (int query_idx = 0; query_idx < kMaxQueries; ++query_idx) {
            pseudo_query_accumulator[query_idx] = 0.0f;
        }

        for (int source_idx = 0; source_idx < num_source_blocks; ++source_idx) {
            float grad_source_accumulator = 0.0f;

            float source_value = 0.0f;
            float inverse_rms_norm = 0.0f;
            float inverse_rms_norm_squared = 0.0f;
            float saved_attention_logit = 0.0f;

            if (valid_bt && valid_hidden) {
                source_value = load_scalar(
                    block_representations +
                    (static_cast<int64_t>(source_idx) * num_batch_seq +
                     batch_seq_idx) *
                        hidden_dim +
                    hidden_idx);
                inverse_rms_norm =
                    inverse_rms_norms[static_cast<int64_t>(batch_seq_idx) *
                                          num_source_blocks +
                                      source_idx];
                inverse_rms_norm_squared = inverse_rms_norm * inverse_rms_norm;
            }

            for (int query_idx = 0; query_idx < num_queries; ++query_idx) {
                float grad_output = 0.0f;
                float pseudo_query_value = 0.0f;

                if (valid_bt && valid_hidden) {
                    grad_output = load_scalar(
                        grad_softmax_outputs +
                        (static_cast<int64_t>(query_idx) * num_batch_seq +
                         batch_seq_idx) *
                            hidden_dim +
                        hidden_idx);
                    pseudo_query_value = load_scalar(
                        pseudo_queries +
                        static_cast<int64_t>(query_idx) * hidden_dim +
                        hidden_idx);
                    saved_attention_logit =
                        attention_logits
                            [static_cast<int64_t>(query_idx) * num_batch_seq *
                                 num_source_blocks +
                             static_cast<int64_t>(batch_seq_idx) *
                                 num_source_blocks +
                             source_idx];
                }

                const float probability =
                    softmax_probabilities
                        [(query_idx * kBlockBt + local_bt) * num_source_blocks +
                         source_idx];
                const float grad_logit =
                    grad_attention_logits
                        [(query_idx * kBlockBt + local_bt) * num_source_blocks +
                         source_idx];

                grad_source_accumulator +=
                    probability * grad_output +
                    grad_logit *
                        (inverse_rms_norm * pseudo_query_value -
                         saved_attention_logit * inverse_rms_norm_squared *
                             source_value / static_cast<float>(hidden_dim));

                pseudo_query_accumulator[query_idx] +=
                    grad_logit * inverse_rms_norm * source_value;
            }

            if (valid_bt && valid_hidden) {
                float grad_source = grad_source_accumulator;
                const int64_t grad_block_offset =
                    (static_cast<int64_t>(source_idx) * num_batch_seq +
                     batch_seq_idx) *
                        hidden_dim +
                    hidden_idx;

                if (accumulate_grad_blocks) {
                    grad_source += load_scalar(
                        grad_block_representations + grad_block_offset);
                }

                store_scalar(
                    grad_block_representations + grad_block_offset,
                    grad_source);
            }
        }

        for (int query_idx = 0; query_idx < num_queries; ++query_idx) {
            pseudo_query_scratch[local_bt * kBlockX + lane] =
                (valid_bt && valid_hidden) ? pseudo_query_accumulator[query_idx]
                                           : 0.0f;
            __syncthreads();

            if (local_bt == 0 && valid_hidden) {
                float tile_sum = 0.0f;
                for (int row = 0; row < kBlockBt; ++row) {
                    tile_sum += pseudo_query_scratch[row * kBlockX + lane];
                }
                atomicAdd(
                    grad_pseudo_queries +
                        static_cast<int64_t>(query_idx) * hidden_dim + hidden_idx,
                    tile_sum);
            }
            __syncthreads();
        }
    }
}

template <typename GradOutT, typename GradBlockT>
void launch_phase_1_backward(
    const __nv_bfloat16* block_representations,
    const __nv_bfloat16* pseudo_queries,
    const float* lses,
    const float* inverse_rms_norms,
    const float* attention_logits,
    const GradOutT* grad_softmax_outputs,
    const float* grad_lses,
    GradBlockT* grad_block_representations,
    float* grad_pseudo_queries,
    int num_source_blocks,
    int num_batch_seq,
    int hidden_dim,
    int num_queries,
    bool has_grad_lses,
    bool accumulate_grad_blocks,
    cudaStream_t stream) {
    const dim3 block(kBlockX, kBlockBt);
    const dim3 grid((num_batch_seq + kBlockBt - 1) / kBlockBt);
    const size_t shared_bytes =
        (kBlockBt * num_source_blocks * kBlockX +
         2 * num_queries * kBlockBt * num_source_blocks +
         kBlockBt * kBlockX) *
        sizeof(float);

    phase_1_backward_kernel<GradOutT, GradBlockT>
        <<<grid, block, shared_bytes, stream>>>(
        block_representations,
        pseudo_queries,
        lses,
        inverse_rms_norms,
        attention_logits,
        grad_softmax_outputs,
        grad_lses,
        grad_block_representations,
        grad_pseudo_queries,
        num_source_blocks,
        num_batch_seq,
        hidden_dim,
        num_queries,
        has_grad_lses,
        accumulate_grad_blocks);
}

void check_contiguous(const torch::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void check_cuda(const torch::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
}

}  // namespace

void phase_1_backward_cuda(
    torch::Tensor block_representations,
    torch::Tensor pseudo_queries,
    torch::Tensor lses,
    torch::Tensor inverse_rms_norms,
    torch::Tensor attention_logits,
    torch::Tensor grad_softmax_outputs,
    torch::Tensor grad_lses,
    torch::Tensor grad_block_representations,
    torch::Tensor grad_pseudo_queries,
    bool has_grad_lses,
    bool accumulate_grad_blocks) {
    check_cuda(block_representations, "block_representations");
    check_cuda(pseudo_queries, "pseudo_queries");
    check_cuda(lses, "lses");
    check_cuda(inverse_rms_norms, "inverse_rms_norms");
    check_cuda(attention_logits, "attention_logits");
    check_cuda(grad_softmax_outputs, "grad_softmax_outputs");
    check_cuda(grad_lses, "grad_lses");
    check_cuda(grad_block_representations, "grad_block_representations");
    check_cuda(grad_pseudo_queries, "grad_pseudo_queries");

    check_contiguous(block_representations, "block_representations");
    check_contiguous(pseudo_queries, "pseudo_queries");
    check_contiguous(lses, "lses");
    check_contiguous(inverse_rms_norms, "inverse_rms_norms");
    check_contiguous(attention_logits, "attention_logits");
    check_contiguous(grad_softmax_outputs, "grad_softmax_outputs");
    check_contiguous(grad_lses, "grad_lses");
    check_contiguous(grad_block_representations, "grad_block_representations");
    check_contiguous(grad_pseudo_queries, "grad_pseudo_queries");

    TORCH_CHECK(
        block_representations.scalar_type() == at::kBFloat16,
        "block_representations must be bf16");
    TORCH_CHECK(
        pseudo_queries.scalar_type() == at::kBFloat16,
        "pseudo_queries must be bf16");
    TORCH_CHECK(lses.scalar_type() == at::kFloat, "lses must be fp32");
    TORCH_CHECK(
        inverse_rms_norms.scalar_type() == at::kFloat,
        "inverse_rms_norms must be fp32");
    TORCH_CHECK(
        attention_logits.scalar_type() == at::kFloat,
        "attention_logits must be fp32");
    TORCH_CHECK(
        grad_lses.scalar_type() == at::kFloat, "grad_lses must be fp32");
    TORCH_CHECK(
        grad_block_representations.scalar_type() == at::kFloat ||
            grad_block_representations.scalar_type() == at::kBFloat16,
        "grad_block_representations must be fp32 or bf16");
    TORCH_CHECK(
        grad_pseudo_queries.scalar_type() == at::kFloat,
        "grad_pseudo_queries must be fp32");
    TORCH_CHECK(
        grad_softmax_outputs.scalar_type() == at::kBFloat16 ||
            grad_softmax_outputs.scalar_type() == at::kFloat,
        "grad_softmax_outputs must be bf16 or fp32");

    const int num_source_blocks = static_cast<int>(block_representations.size(0));
    const int batch = static_cast<int>(block_representations.size(1));
    const int sequence = static_cast<int>(block_representations.size(2));
    const int hidden_dim = static_cast<int>(block_representations.size(3));
    const int num_batch_seq = batch * sequence;
    const int num_queries = static_cast<int>(pseudo_queries.size(0));

    TORCH_CHECK(
        num_source_blocks <= kMaxSourceBlocks,
        "CUDA phase 1 backward supports at most ",
        kMaxSourceBlocks,
        " source blocks");
    TORCH_CHECK(
        num_queries <= kMaxQueries,
        "CUDA phase 1 backward supports at most ",
        kMaxQueries,
        " queries per call");

    c10::cuda::CUDAGuard device_guard(block_representations.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    const auto* block_ptr = reinterpret_cast<const __nv_bfloat16*>(
        block_representations.data_ptr<at::BFloat16>());
    const auto* query_ptr = reinterpret_cast<const __nv_bfloat16*>(
        pseudo_queries.data_ptr<at::BFloat16>());

    auto launch_with_grad_block = [&](auto grad_output_ptr) {
        if (grad_block_representations.scalar_type() == at::kBFloat16) {
            auto* grad_block_ptr = reinterpret_cast<__nv_bfloat16*>(
                grad_block_representations.data_ptr<at::BFloat16>());
            launch_phase_1_backward(
                block_ptr,
                query_ptr,
                lses.data_ptr<float>(),
                inverse_rms_norms.data_ptr<float>(),
                attention_logits.data_ptr<float>(),
                grad_output_ptr,
                grad_lses.data_ptr<float>(),
                grad_block_ptr,
                grad_pseudo_queries.data_ptr<float>(),
                num_source_blocks,
                num_batch_seq,
                hidden_dim,
                num_queries,
                has_grad_lses,
                accumulate_grad_blocks,
                stream);
        } else {
            launch_phase_1_backward(
                block_ptr,
                query_ptr,
                lses.data_ptr<float>(),
                inverse_rms_norms.data_ptr<float>(),
                attention_logits.data_ptr<float>(),
                grad_output_ptr,
                grad_lses.data_ptr<float>(),
                grad_block_representations.data_ptr<float>(),
                grad_pseudo_queries.data_ptr<float>(),
                num_source_blocks,
                num_batch_seq,
                hidden_dim,
                num_queries,
                has_grad_lses,
                accumulate_grad_blocks,
                stream);
        }
    };

    if (grad_softmax_outputs.scalar_type() == at::kBFloat16) {
        const auto* grad_output_ptr = reinterpret_cast<const __nv_bfloat16*>(
            grad_softmax_outputs.data_ptr<at::BFloat16>());
        launch_with_grad_block(grad_output_ptr);
    } else {
        launch_with_grad_block(grad_softmax_outputs.data_ptr<float>());
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
