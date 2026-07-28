# Developer tools -- not shipped, not linked into toby_server.
#
# pretokenize_dump is the C++ half of the differential test in
# tools/pretokenizer_diff.py: the Python side owns the reference regex, this side
# runs the real pretokenizer. See the comment at the top of either file.

add_executable(pretokenize_dump pretokenize_dump.cpp)

target_link_libraries(pretokenize_dump
  PRIVATE
    toby::tokenize
    toby::warnings
    toby::sanitizers
    toby::simd)

toby_enable_tidy(pretokenize_dump)

# vocab_dump prints what the vocabulary loaders parsed out of a real file.
# Pair it with tools/fetch_vocab.py, which downloads the files to point it at.
add_executable(vocab_dump vocab_dump.cpp)

target_link_libraries(vocab_dump
  PRIVATE
    toby::tokenize
    toby::warnings
    toby::sanitizers
    toby::simd)

toby_enable_tidy(vocab_dump)
