# tokenize/ -- the tokenizer (pre-tokenizer + BPE). This is NOT the protected
# core/ zone: it's yours to write, and the assistant may scaffold tests + build
# wiring here. The interesting algorithm work still lives with you.

add_library(toby_tokenize
  pretokenizer.cpp)
add_library(toby::tokenize ALIAS toby_tokenize)

target_include_directories(toby_tokenize
  PUBLIC
    ${CMAKE_SOURCE_DIR}/include)

# Strict warnings + sanitizers apply to this code too (PRIVATE: consumers get
# the include dir but not our flags).
target_link_libraries(toby_tokenize
  PRIVATE
    toby::warnings
    toby::sanitizers
    toby::simd)

toby_enable_tidy(toby_tokenize)
