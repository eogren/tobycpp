# Defines an INTERFACE target `toby_warnings` carrying a strict, curated warning
# set. Link it into first-party targets only (PRIVATE), never into third-party
# dependencies.

add_library(toby_warnings INTERFACE)
add_library(toby::warnings ALIAS toby_warnings)

set(
  _toby_common_warnings
  -Wall
  -Wextra # reasonable and standard extras
  -Wpedantic # warn on non-standard C++
  -Wshadow # a variable declaration shadows a parent scope
  -Wnon-virtual-dtor # class with virtuals but non-virtual dtor
  -Wold-style-cast # C-style casts
  -Wcast-align # potential performance-costly casts
  -Wunused # anything unused
  -Woverloaded-virtual # overload (not override) of a virtual
  -Wconversion # implicit conversions that may alter a value
  -Wsign-conversion # implicit sign conversions
  -Wnull-dereference # a null dereference is detected
  -Wdouble-promotion # float implicitly promoted to double
  -Wformat=2 # security-relevant printf/scanf format checks
  -Wimplicit-fallthrough # missing break in a switch
)

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  set(_toby_warnings ${_toby_common_warnings})
else()
  message(WARNING "toby: no curated warning flags for '${CMAKE_CXX_COMPILER_ID}'.")
  set(_toby_warnings ${_toby_common_warnings})
endif()

if(TOBY_WARNINGS_AS_ERRORS)
  list(APPEND _toby_warnings -Werror)
endif()

target_compile_options(toby_warnings INTERFACE ${_toby_warnings})
