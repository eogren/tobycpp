add_library(gpt2_kernels equality.cpp)
add_library(toby::gpt2_kernels ALIAS gpt2_kernels)

if(TOBY_ENABLE_CUDA)
  target_sources(gpt2_kernels PRIVATE kernels.cu)
endif()

target_include_directories(gpt2_kernels PUBLIC ${CMAKE_SOURCE_DIR}/include})
target_link_libraries(gpt2_kernels PUBLIC toby::safetensors)
