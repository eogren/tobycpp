# tokenize/ -- the tokenizer (pre-tokenizer + BPE). This is NOT the protected
# core/ zone: it's yours to write, and the assistant may scaffold tests + build
# wiring here. The interesting algorithm work still lives with you.

add_library(toby_tokenize
  pretokenizer.cpp
  vocab.cpp)
add_library(toby::tokenize ALIAS toby_tokenize)

target_include_directories(toby_tokenize
  PUBLIC
    ${CMAKE_SOURCE_DIR}/include)

# Strict warnings + sanitizers apply to this code too (PRIVATE: consumers get
# the include dir but not our flags).
#
# nlohmann_json is PRIVATE on purpose: vocab.hpp exposes only std types, so the
# JSON library stays an implementation detail of vocab.cpp and never leaks into
# a consumer's include path. That is what makes swapping it a local change.
target_link_libraries(toby_tokenize
  PRIVATE
    nlohmann_json::nlohmann_json
    toby::warnings
    toby::sanitizers
    toby::simd)

toby_enable_tidy(toby_tokenize)
