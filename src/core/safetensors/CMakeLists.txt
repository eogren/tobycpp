# tokenize/ -- the tokenizer (pre-tokenizer + BPE). This is NOT the protected
# core/ zone: it's yours to write, and the assistant may scaffold tests + build
# wiring here. The interesting algorithm work still lives with you.

add_library(
  toby_safetensors
  arena.cpp
  cpu_arena.cpp
  file.cpp
  safetensors.cpp
)
add_library(toby::safetensors ALIAS toby_safetensors)

target_include_directories(toby_safetensors PUBLIC ${CMAKE_SOURCE_DIR}/include)

# Strict warnings + sanitizers apply to this code too (PRIVATE: consumers get
# the include dir but not our flags).
#
# nlohmann_json is PRIVATE on purpose: vocab.hpp exposes only std types, so the
# JSON library stays an implementation detail of vocab.cpp and never leaks into
# a consumer's include path. That is what makes swapping it a local change.
#
# ICU::uc is PRIVATE for the same reason: UTF-8 traversal and Unicode property
# lookup are tokenizer implementation details. Keep ICU types out of the public
# headers so consumers do not inherit the dependency.
#
# spdlog::spdlog is PRIVATE too: logging calls are internal to this library's
# implementation, not part of its public API.
#
target_link_libraries(
  toby_safetensors
  PRIVATE
    nlohmann_json::nlohmann_json
    spdlog::spdlog
    toby::warnings
    toby::sanitizers
    toby::simd
)

toby_enable_tidy(toby_safetensors)
