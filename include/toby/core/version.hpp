#ifndef TOBY_CORE_VERSION_HPP
#define TOBY_CORE_VERSION_HPP

#include <string_view>

namespace toby::core {

/// Returns the library version string, e.g. "0.1.0".
///
/// Placeholder to exercise the build/test pipeline. This header lives in the
/// PROTECTED learning zone (include/toby/core/) -- it is yours to grow into the
/// real inference engine API (tensors, ops, model, session, ...).
[[nodiscard]] std::string_view library_version() noexcept;

} // namespace toby::core

#endif // TOBY_CORE_VERSION_HPP
