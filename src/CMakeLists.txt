# First-party source lives here. Two layers:
#   core/  -> the inference engine you are writing to learn (PROTECTED zone).
#   this   -> the server/executable that wires the core into a running process.

add_subdirectory(core)
add_subdirectory(tokenize)

# ---------------------------------------------------------------------------
# toby_server: the executable entry point.
# This is "plumbing" the assistant may help scaffold. The learning lives in
# toby::core, which this links against.
# ---------------------------------------------------------------------------
add_executable(toby_server main.cpp)
add_executable(toby::server ALIAS toby_server)

target_link_libraries(toby_server
  PRIVATE
    toby::core
    httplib::httplib
    toby::warnings
    toby::sanitizers
    toby::simd)

toby_enable_tidy(toby_server)
