#include "toby/core/version.hpp"

namespace toby::core {

std::string_view library_version() noexcept {
    // TOBY_VERSION is injected by CMake below; falls back if built by hand.
#ifdef TOBY_VERSION
    return TOBY_VERSION;
#else
    return "0.0.0-dev";
#endif
}

} // namespace toby::core
