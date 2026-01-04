#include <windows.h>
#include <stdio.h>
#include <cstdlib>

#include "server.hpp"
#include "api_handler.hpp"
#include "config.hpp"

void init_console() {
    SetConsoleCP(CP_UTF8);
    SetConsoleOutputCP(CP_UTF8);
    fprintf(stdout, "[INIT] Console initialized\n");
    fflush(stdout);
}

int main(int argc, char* argv[]) {
    (void)argc;
    (void)argv;

    init_console();

    printf("[MAIN] Starting server...\n");
    fflush(stdout);

    try {
        printf("[MAIN] Creating HTTPServer object...\n");
        fflush(stdout);

        HTTPServer server;

        printf("[MAIN] Registering endpoints...\n");
        fflush(stdout);

        server.registerEndpoint("POST", "/api/process", handleGraphProcessing);
        server.registerEndpoint("POST", "/api/search", handleSearchAnswer);
        server.registerEndpoint("GET", "/api/types", handleGetGraphTypes);
        server.registerEndpoint("GET", "/api/health", handleHealth);
        server.registerEndpoint("GET", "/api/version", handleVersion);
        
        server.registerEndpoint("OPTIONS", "/api/process", [](const Request& req) {
            (void)req;
            return R"({"status":"ok"})";
        });

        server.registerEndpoint("OPTIONS", "/api/search", [](const Request& req) {
            (void)req;
            return R"({"status":"ok"})";
        });

        printf("[MAIN] Setting up static files...\n");
        fflush(stdout);

        server.serveStatic("/", "./web/");
        server.serveStatic("/temp/", "./temp/");

        printf("[MAIN] Configuration complete\n");
        fflush(stdout);

        printf("[MAIN] Starting listener on %s:%d\n", Config::HOST, Config::PORT);
        fflush(stdout);

        server.listen(Config::PORT, Config::HOST);

        printf("[MAIN] Listener exited (should never reach here)\n");
        fflush(stdout);

        return 0;

    } catch (const std::exception& e) {
        printf("[ERROR] Exception: %s\n", e.what());
        fflush(stdout);
        fprintf(stderr, "[ERROR] Exception: %s\n", e.what());
        fflush(stderr);
        return 1;

    } catch (...) {
        printf("[ERROR] Unknown exception\n");
        fflush(stdout);
        fprintf(stderr, "[ERROR] Unknown exception\n");
        fflush(stderr);
        return 1;
    }
}