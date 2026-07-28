#include "toby/core/version.hpp"

#include <cstdio>
#include <exception>
#include <httplib.h>
#include <print>

// Entry point / server bootstrap. This is plumbing, not the core learning
// material -- it exists to give you a running process to hang the engine off of.
//
// For now it exposes a single liveness endpoint, GET /healthz, so a process
// manager / load balancer has something to poll. Real request handling
// (completions, streaming, etc.) hangs off toby::core once the engine exists.
int main() {
    try {
        constexpr const char* host = "127.0.0.1";
        constexpr int port = 8080;

        httplib::Server svr;

        // Liveness probe: the process is up and the HTTP layer is serving.
        // (This says nothing about the engine being ready -- add a /readyz that
        // checks toby::core once there's an engine to check.)
        svr.Get("/healthz", [](const httplib::Request&, httplib::Response& res) {
            res.set_content("ok\n", "text/plain");
        });

        std::println("toby inference server {}", toby::core::library_version());
        std::println("listening on http://{}:{}  (GET /healthz)", host, port);

        if (!svr.listen(host, port)) {
            std::println(stderr, "error: failed to bind {}:{}", host, port);
            return 1;
        }
        return 0;
    } catch (const std::exception& error) {
        std::fputs("fatal error: ", stderr);
        std::fputs(error.what(), stderr);
        std::fputc('\n', stderr);
        return 1;
    } catch (...) {
        std::fputs("fatal error: unknown exception\n", stderr);
        return 1;
    }
}
