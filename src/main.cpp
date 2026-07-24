#include "toby/core/version.hpp"

#include <print>

// Entry point / server bootstrap. This is plumbing, not the core learning
// material -- it exists to give you a running process to hang the engine off of.
int main() {
    std::println("toby inference server {}", toby::core::library_version());
    std::println("(skeleton -- wire up the engine in toby::core)");
    return 0;
}
