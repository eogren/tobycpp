#include "toby/core/version.hpp"

#include <catch2/catch_test_macros.hpp>

// A smoke test proving the build -> link -> test pipeline works end to end.
// As you build the engine in toby::core, add real tests alongside it. Tests
// live OUTSIDE the protected zone, so the assistant can help you scaffold and
// expand these -- but let the assertions reflect behavior you actually intend.
TEST_CASE("library_version is non-empty and looks like a version", "[core][version]") {
    const auto version = toby::core::library_version();

    REQUIRE_FALSE(version.empty());
    // Expect something of the form "N.N.N".
    CHECK(version.find('.') != std::string_view::npos);
}
