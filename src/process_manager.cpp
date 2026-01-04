#include "process_manager.hpp"
#include "file_manager.hpp"
#include "config.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>
#include <thread>

#ifdef _WIN32
#include <windows.h>
#include <process.h>
#else
#include <unistd.h>
#include <sys/wait.h>
#include <signal.h>
#endif

using json = nlohmann::json;

bool runGraphProcessing(
    const std::string& text,
    const std::string& graph_type,
    std::string& error_msg,
    json& graph_data,
    std::string& csv_file,
    std::string& html_file
) {
    try {
        std::cout << "\n[PM] ===== Graph Processing Started =====" << std::endl;
        std::cout << "[PM] Graph Type: " << graph_type << std::endl;
        std::cout << "[PM] Text Length: " << text.length() << " bytes" << std::endl;

        auto now = std::chrono::system_clock::now();
        auto time_t_now = std::chrono::system_clock::to_time_t(now);
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            now.time_since_epoch()
        ) % 1000;

        std::string session_id = std::to_string(time_t_now) + "_" + std::to_string(ms.count());
        
        csv_file = std::string(Config::TEMP_DIR) + "graph_" + session_id + ".csv";
        html_file = std::string(Config::TEMP_DIR) + "graph_" + session_id + ".html";
        std::string input_json_file = std::string(Config::TEMP_DIR) + "input_" + session_id + ".json";
        std::string py_stdout_file = std::string(Config::TEMP_DIR) + "py_stdout_" + session_id + ".txt";
        std::string py_stderr_file = std::string(Config::TEMP_DIR) + "py_stderr_" + session_id + ".txt";

        std::cout << "[PM] Session ID: " << session_id << std::endl;
        std::cout << "[PM] Output CSV: " << csv_file << std::endl;
        std::cout << "[PM] Output HTML: " << html_file << std::endl;

        if (!FileManager::createDirectory(Config::TEMP_DIR)) {
            error_msg = "Failed to create temp directory: " + std::string(Config::TEMP_DIR);
            std::cout << "[PM] ✗ " << error_msg << std::endl;
            return false;
        }

        std::cout << "[PM] ✓ Temp directory ready" << std::endl;

        json request;
        request["text"] = text;
        request["graph_type"] = graph_type;
        request["max_nodes"] = Config::MAX_GRAPH_NODES;
        request["min_word_length"] = Config::MIN_WORD_LENGTH;
        request["output_csv"] = csv_file;
        request["output_html"] = html_file;

        std::string request_json = request.dump();
        std::cout << "[PM] Request prepared (" << request_json.length() << " bytes)" << std::endl;

        std::cout << "[PM] Writing JSON to file: " << input_json_file << std::endl;
        try {
            std::ofstream json_file(input_json_file);
            if (!json_file.is_open()) {
                error_msg = "Failed to create input JSON file";
                std::cout << "[PM] ✗ " << error_msg << std::endl;
                return false;
            }
            json_file << request_json;
            json_file.close();
            std::cout << "[PM] ✓ JSON file created" << std::endl;
        } catch (const std::exception& e) {
            error_msg = "Failed to write JSON file: " + std::string(e.what());
            std::cout << "[PM] ✗ " << error_msg << std::endl;
            return false;
        }

        std::string script_name;
        if (graph_type == "syntax") {
            script_name = "graph_builder_syntax.py";
        } else if (graph_type == "semantic") {
            script_name = "graph_builder_semantic.py";
        } else if (graph_type == "hybrid") {
            script_name = "graph_builder_hybrid.py";
        } else {
            error_msg = "Unknown graph type: " + graph_type;
            return false;
        }

        std::string script_path = std::string(Config::PYTHON_SCRIPT_DIR) + script_name;

        std::ifstream test_file(script_path);
        if (!test_file.good()) {
            error_msg = "Python script not found: " + script_path;
            std::cout << "[PM] ✗ Script not found at: " << script_path << std::endl;
            return false;
        }
        test_file.close();
        std::cout << "[PM] ✓ Using script: " << script_path << std::endl;

        std::string python_cmd;

#ifdef _WIN32
        python_cmd = "python \"" + script_path + "\" \"" + input_json_file + "\" > \"" + 
                     py_stdout_file + "\" 2> \"" + py_stderr_file + "\"";
#else
        // Linux/Mac: python3
        python_cmd = "python3 \"" + script_path + "\" \"" + input_json_file + "\" > \"" + 
                     py_stdout_file + "\" 2> \"" + py_stderr_file + "\"";
#endif

        std::cout << "[PM] Command: " << python_cmd << std::endl;

#ifdef _WIN32
        PROCESS_INFORMATION piProcInfo;
        STARTUPINFOA siStartInfo;
        
        ZeroMemory(&piProcInfo, sizeof(PROCESS_INFORMATION));
        ZeroMemory(&siStartInfo, sizeof(STARTUPINFOA));
        siStartInfo.cb = sizeof(STARTUPINFOA);

        BOOL bSuccess = CreateProcessA(
            NULL,
            (LPSTR)python_cmd.c_str(),
            NULL, NULL,
            TRUE,
            0,
            NULL,
            NULL,
            &siStartInfo,
            &piProcInfo
        );

        if (!bSuccess) {
            error_msg = "Failed to create Python process (error: " + std::to_string(GetLastError()) + ")";
            std::cout << "[PM] ✗ CreateProcessA failed" << std::endl;
            return false;
        }

        std::cout << "[PM] ✓ Python process created (PID: " << piProcInfo.dwProcessId << ")" << std::endl;

        DWORD dwWaitResult = WaitForSingleObject(piProcInfo.hProcess, Config::PROCESS_TIMEOUT_SEC * 1000);

        if (dwWaitResult == WAIT_TIMEOUT) {
            TerminateProcess(piProcInfo.hProcess, 1);
            CloseHandle(piProcInfo.hProcess);
            CloseHandle(piProcInfo.hThread);
            error_msg = "Python process timeout (>" + std::to_string(Config::PROCESS_TIMEOUT_SEC) + "s)";
            std::cout << "[PM] ✗ Process timeout" << std::endl;
            return false;
        }

        DWORD dwExitCode = 0;
        GetExitCodeProcess(piProcInfo.hProcess, &dwExitCode);
        CloseHandle(piProcInfo.hProcess);
        CloseHandle(piProcInfo.hThread);

        std::cout << "[PM] Python exit code: " << dwExitCode << std::endl;

        if (dwExitCode != 0) {
            std::string stderr_content = "";
            if (FileManager::fileExists(py_stderr_file)) {
                try {
                    stderr_content = FileManager::readFile(py_stderr_file);
                    std::cout << "\n[PM] ===== Python stderr =====" << std::endl;
                    std::cout << stderr_content << std::endl;
                } catch (...) {}
            }

            if (FileManager::fileExists(py_stdout_file)) {
                try {
                    std::string stdout_content = FileManager::readFile(py_stdout_file);
                    std::cout << "\n[PM] ===== Python stdout =====" << std::endl;
                    std::cout << stdout_content << std::endl;
                } catch (...) {}
            }

            error_msg = "Python process exited with code " + std::to_string(dwExitCode);
            if (!stderr_content.empty()) {
                error_msg += "\nSTDERR: " + stderr_content;
            }

            std::cout << "[PM] ✗ " << error_msg << std::endl;
            return false;
        }

#else
        // ============ LINUX/MAC VERSION ============
        pid_t pid = fork();

        if (pid == -1) {
            error_msg = "Failed to fork process";
            return false;
        }

        if (pid == 0) {
            // Child process
            system(python_cmd.c_str());
            exit(0);
        } else {
            // Parent process
            int status;
            auto start = std::chrono::steady_clock::now();

            while (true) {
                pid_t result = waitpid(pid, &status, WNOHANG);
                
                if (result == pid) break;
                
                if (result == -1) {
                    error_msg = "waitpid failed";
                    return false;
                }

                auto now = std::chrono::steady_clock::now();
                if (std::chrono::duration_cast<std::chrono::seconds>(now - start).count() > 
                    Config::PROCESS_TIMEOUT_SEC) {
                    kill(pid, SIGKILL);
                    error_msg = "Python process timeout";
                    std::cout << "[PM] ✗ Process timeout" << std::endl;
                    return false;
                }

                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }

            if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
                std::string stderr_content = "";
                if (FileManager::fileExists(py_stderr_file)) {
                    try {
                        stderr_content = FileManager::readFile(py_stderr_file);
                        std::cout << "\n[PM] ===== Python stderr =====" << std::endl;
                        std::cout << stderr_content << std::endl;
                    } catch (...) {}
                }

                error_msg = "Python process failed";
                if (!stderr_content.empty()) {
                    error_msg += "\n" + stderr_content;
                }

                std::cout << "[PM] ✗ " << error_msg << std::endl;
                return false;
            }
        }
#endif

        std::cout << "[PM] ✓ Python process completed successfully" << std::endl;
        std::cout << "[PM] Checking output files..." << std::endl;

        if (!FileManager::fileExists(csv_file)) {
            error_msg = "CSV file not created: " + csv_file;
            std::cout << "[PM] ✗ CSV not found" << std::endl;
            
            if (FileManager::fileExists(py_stderr_file)) {
                try {
                    std::string stderr_content = FileManager::readFile(py_stderr_file);
                    std::cout << "[PM] Python stderr:" << std::endl << stderr_content << std::endl;
                    error_msg += "\n" + stderr_content;
                } catch (...) {}
            }

            return false;
        }

        if (!FileManager::fileExists(html_file)) {
            error_msg = "HTML file not created: " + html_file;
            std::cout << "[PM] ✗ HTML not found" << std::endl;
            return false;
        }

        std::cout << "[PM] ✓ Output files created" << std::endl;

        std::string csv_content = FileManager::readFile(csv_file);

        graph_data["csv_file"] = csv_file;
        graph_data["html_file"] = html_file;
        graph_data["node_count"] = 0;
        graph_data["edge_count"] = 0;
        graph_data["type"] = graph_type;

        int edge_count = 0;
        int line_count = 0;
        for (char c : csv_content) {
            if (c == '\n') line_count++;
        }

        edge_count = std::max(0, line_count - 2);
        graph_data["edge_count"] = edge_count;

        std::cout << "[PM] ✓ Results parsed successfully" << std::endl;
        std::cout << "[PM] Edge count: " << edge_count << std::endl;
        std::cout << "[PM] ===== Graph Processing Completed =====" << std::endl << std::endl;

        try {
            if (FileManager::fileExists(input_json_file)) {
                std::remove(input_json_file.c_str());
            }
            if (FileManager::fileExists(py_stdout_file)) {
                std::remove(py_stdout_file.c_str());
            }
            if (FileManager::fileExists(py_stderr_file)) {
                std::remove(py_stderr_file.c_str());
            }
        } catch (...) {
        }

        return true;

    } catch (const std::exception& e) {
        error_msg = std::string("Exception: ") + e.what();
        std::cerr << "[PM] Error: " << error_msg << std::endl;
        return false;
    }
}