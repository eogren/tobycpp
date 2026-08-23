#include "toby/safetensors/align.hpp"

#include <catch2/catch_test_macros.hpp>

TEST_CASE("align_up rounds up to the next multiple", "[align]") {
    CHECK(toby::tensors::align_up(0, 64) == 0);
    CHECK(toby::tensors::align_up(1, 64) == 64);
    CHECK(toby::tensors::align_up(63, 64) == 64);
    CHECK(toby::tensors::align_up(64, 64) == 64);
    CHECK(toby::tensors::align_up(65, 64) == 128);
}

TEST_CASE("align_up works for non-power-of-two alignments", "[align]") {
    CHECK(toby::tensors::align_up(10, 10) == 10);
    CHECK(toby::tensors::align_up(11, 10) == 20);
    CHECK(toby::tensors::align_up(20, 10) == 20);
}

TEST_CASE("align_up with alignment of 1 is a no-op", "[align]") {
    CHECK(toby::tensors::align_up(0, 1) == 0);
    CHECK(toby::tensors::align_up(7, 1) == 7);
}
