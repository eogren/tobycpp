#include "support/temp_file.hpp"
#include "toby/safetensors/file.hpp"

#include <catch2/catch_test_macros.hpp>

using toby::tensors::open_file;
using toby::test::TempFile;

namespace {
TempFile standard_fixture() {
    return TempFile{"Hello world"};
}
} // namespace

TEST_CASE("test open file", "[file]") {
    auto fixture = standard_fixture();
    auto fd = open_file(fixture.path());
}
