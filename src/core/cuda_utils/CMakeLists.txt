add_library(toby_cuda_utils cuda_alloc.cpp)
add_library(toby::cuda_utils ALIAS toby_cuda_utils)

target_include_directories(toby_cuda_utils PUBLIC ${CMAKE_SOURCE_DIR}/include)

target_link_libraries(
  toby_cuda_utils
  PRIVATE
    toby::cuda
    toby::warnings
    toby::sanitizers
)
target_link_libraries(toby_cuda_utils PUBLIC spdlog::spdlog)

toby_enable_tidy(toby_cuda_utils)
