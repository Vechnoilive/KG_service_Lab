#ifndef CONFIG_HPP
#define CONFIG_HPP

namespace Config {
    static constexpr const char* VERSION = "1.4";
    static constexpr int PORT = 8080;
    static constexpr const char* HOST = "0.0.0.0";
    static constexpr const char* TEMP_DIR = "./temp/";
    static constexpr const char* PYTHON_SCRIPT_DIR = "./python/";
    static constexpr const char* WEB_DIR = "./web/";
    static constexpr int PROCESS_TIMEOUT_SEC = 300;
    static constexpr size_t MAX_BODY_SIZE = 10 * 1024 * 1024;
    static constexpr size_t MAX_TEXT_LENGTH = 10000;
    static constexpr int MAX_GRAPH_NODES = 1000;
    static constexpr int MIN_WORD_LENGTH = 2;
}

#endif