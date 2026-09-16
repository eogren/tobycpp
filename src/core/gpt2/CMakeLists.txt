add_library(gpt2_kernels kernels.cu)
add_library(toby::gpt2_kernels ALIAS gpt2_kernels)

target_link_libraries(gpt2_kernels PUBLIC toby::safetensors)
