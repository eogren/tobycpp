# =============================================================================
# PROTECTED LEARNING ZONE  --  src/core/
#
# This library is the part of the project you are writing yourself in order to
# learn. Per CLAUDE.md, the assistant must not implement or edit files under
# src/core/ or include/toby/core/. It may help you reason about them via
# questions and hints, but the code here is yours.
#
# The single source file below is a placeholder that only proves the toolchain
# works end to end (build -> link -> test). Replace it as you build the engine.
# =============================================================================

add_library(toby_core version.cpp)
add_library(toby::core ALIAS toby_core)

target_include_directories(toby_core PUBLIC ${CMAKE_SOURCE_DIR}/include)

# Surface the project version to the source as a string literal.
target_compile_definitions(toby_core PRIVATE TOBY_VERSION="${PROJECT_VERSION}")

# Strict warnings + sanitizers apply to your code too. Note these are PRIVATE:
# consumers of toby::core inherit the include dir but not our warning flags.
target_link_libraries(
  toby_core
  PRIVATE
    toby::warnings
    toby::sanitizers
)

toby_enable_tidy(toby_core)
