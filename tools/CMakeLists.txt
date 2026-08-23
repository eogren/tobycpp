# Developer tools -- not shipped, not linked into toby_server.
#
# pretokenize_dump is the C++ half of the differential test in
# tools/pretokenizer_diff.py: the Python side owns the reference regex, this side
# runs the real pretokenizer. See the comment at the top of either file.

add_executable(pretokenize_dump pretokenize_dump.cpp)

target_link_libraries(
  pretokenize_dump
  PRIVATE
    toby::tokenize
    toby::warnings
    toby::sanitizers
    toby::simd
)

toby_enable_tidy(pretokenize_dump)

# vocab_dump prints what the vocabulary loaders parsed out of a real file.
# Pair it with tools/fetch_vocab.py, which downloads the files to point it at.
add_executable(vocab_dump vocab_dump.cpp)

target_include_directories(vocab_dump PRIVATE ${CMAKE_SOURCE_DIR}/src/tokenize)

target_link_libraries(
  vocab_dump
  PRIVATE
    toby::tokenize
    toby::warnings
    toby::sanitizers
    toby::simd
)

toby_enable_tidy(vocab_dump)

add_executable(bpe_dump bpe_dump.cpp)

target_compile_definitions(bpe_dump PRIVATE TOBY_VOCAB_DIR="${CMAKE_SOURCE_DIR}/vocab")

target_link_libraries(
  bpe_dump
  PRIVATE
    toby::tokenize
    toby::warnings
    toby::sanitizers
    toby::simd
)

toby_enable_tidy(bpe_dump)

# cuda_probe answers "is the GPU side of this machine actually usable" before any
# engine code depends on it. Only built with -DTOBY_ENABLE_CUDA=ON, since without
# it toby::cuda is a stub with no runtime behind it.
if(TOBY_ENABLE_CUDA)
  add_executable(cuda_probe cuda_probe.cpp)

  target_link_libraries(
    cuda_probe
    PRIVATE
      toby::cuda
      toby::warnings
      toby::sanitizers
      toby::simd
  )

  toby_enable_tidy(cuda_probe)
endif()
