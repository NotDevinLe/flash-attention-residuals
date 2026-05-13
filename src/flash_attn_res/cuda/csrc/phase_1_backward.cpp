#include <torch/extension.h>

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
    bool accumulate_grad_blocks);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("phase_1_backward", &phase_1_backward_cuda, "Phase 1 backward CUDA");
}

