#include "toby/core/version.hpp"

#include <exception>
#include <httplib.h>
#include <spdlog/cfg/env.h>
#include <spdlog/spdlog.h>

// Entry point / server bootstrap. This is plumbing, not the core learning
// material -- it exists to give you a running process to hang the engine off of.
//
// For now it exposes a single liveness endpoint, GET /healthz, so a process
// manager / load balancer has something to poll. Real request handling
// (completions, streaming, etc.) hangs off toby::core once the engine exists.
int main() {
    // Default level is "info"; override per-run with e.g. `SPDLOG_LEVEL=debug ./toby_server`.
    spdlog::cfg::load_env_levels();

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

        spdlog::info("toby inference server {}", toby::core::library_version());
        spdlog::info("listening on http://{}:{}  (GET /healthz)", host, port);

        if (!svr.listen(host, port)) {
            spdlog::error("failed to bind {}:{}", host, port);
            return 1;
        }
        return 0;
    } catch (const std::exception& error) {
        spdlog::critical("fatal error: {}", error.what());
        return 1;
    } catch (...) {
        spdlog::critical("fatal error: unknown exception");
        return 1;
    }
}
