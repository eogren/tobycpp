# Defines an INTERFACE target `toby_sanitizers` that applies the sanitizers
# selected via cache options to both compile and link steps. Link it into
# first-party targets (PRIVATE). No-op when nothing is enabled.

add_library(toby_sanitizers INTERFACE)
add_library(toby::sanitizers ALIAS toby_sanitizers)

set(_toby_sanitizers "")

if(TOBY_SANITIZE_ADDRESS)
  list(APPEND _toby_sanitizers address)
endif()

if(TOBY_SANITIZE_UNDEFINED)
  list(APPEND _toby_sanitizers undefined)
endif()

if(TOBY_SANITIZE_THREAD)
  if(TOBY_SANITIZE_ADDRESS OR TOBY_SANITIZE_UNDEFINED)
    message(FATAL_ERROR
      "ThreadSanitizer cannot be combined with Address/Undefined sanitizers.")
  endif()
  list(APPEND _toby_sanitizers thread)
endif()

if(_toby_sanitizers)
  if(NOT (CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU"))
    message(WARNING "toby: sanitizers requested but unsupported for this compiler.")
  else()
    list(JOIN _toby_sanitizers "," _toby_sanitizer_arg)
    message(STATUS "toby: sanitizers enabled -> ${_toby_sanitizer_arg}")
    target_compile_options(toby_sanitizers INTERFACE
      -fsanitize=${_toby_sanitizer_arg}
      -fno-omit-frame-pointer
      -g)
    target_link_options(toby_sanitizers INTERFACE
      -fsanitize=${_toby_sanitizer_arg})
  endif()
endif()
