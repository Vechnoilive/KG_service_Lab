#include "api_handler.hpp"
#include "process_manager.hpp"
#include "file_manager.hpp"
#include <filesystem>
#include <algorithm>
#include "config.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <set>
#include <nlohmann/json.hpp>
#include <cstdlib>

using json = nlohmann::json;

std::string decodeBase64(const std::string& encoded) {
    static const std::string base64_chars =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string decoded;
    std::vector<int> T(256, -1);
    for (int i = 0; i < 64; i++) T[base64_chars[i]] = i;

    int val = 0, valb = -6;
    for (unsigned char c : encoded) {
        if (T[c] == -1) break;
        val = (val << 6) + T[c];
        valb += 6;
        if (valb >= 0) {
            decoded.push_back(char((val >> valb) & 0xFF));
            valb -= 8;
        }
    }
    return decoded;
}

std::string trim(const std::string& str) {
    size_t first = str.find_first_not_of(" \t\r\n\0");
    if (first == std::string::npos) return "";
    size_t last = str.find_last_not_of(" \t\r\n\0");
    return str.substr(first, (last - first + 1));
}

json validateText(const std::string& text) {
    json response;
    if (text.empty()) {
        response["status"] = "error";
        response["message"] = "Text cannot be empty";
        response["error_code"] = "EMPTY_TEXT";
        return response;
    }

    if (text.length() > Config::MAX_TEXT_LENGTH) {
        response["status"] = "error";
        response["message"] = "Text is too long. Maximum " + std::to_string(Config::MAX_TEXT_LENGTH) + " characters";
        response["error_code"] = "TEXT_TOO_LONG";
        return response;
    }

    response["status"] = "success";
    return response;
}

json validateGraphType(const std::string& graph_type) {
    json response;
    if (graph_type.empty()) {
        response["status"] = "error";
        response["message"] = "Graph type cannot be empty";
        response["error_code"] = "EMPTY_GRAPH_TYPE";
        return response;
    }

    if (graph_type != "syntax" && graph_type != "semantic" && graph_type != "hybrid") {
        response["status"] = "error";
        response["message"] = "Unknown graph type. Allowed: syntax, semantic, hybrid";
        response["error_code"] = "INVALID_GRAPH_TYPE";
        return response;
    }

    response["status"] = "success";
    return response;
}

json loadEdgesFromCSV(const std::string& csv_path) {
    json edges_array = json::array();
    std::cerr << "[API] Loading edges from CSV: " << csv_path << "\n";
    std::cerr.flush();

    std::ifstream csv_file(csv_path);
    if (!csv_file.is_open()) {
        std::cerr << "[API] ✗ Cannot open CSV file: " << csv_path << "\n";
        std::cerr.flush();
        return edges_array;
    }

    std::string line;
    bool first_line = true;
    int edge_count = 0;

    while (std::getline(csv_file, line)) {
        if (first_line) {
            first_line = false;
            continue;
        }

        std::istringstream ss(line);
        std::string source, target, relation, weight_str;

        if (std::getline(ss, source, ',') &&
            std::getline(ss, target, ',') &&
            std::getline(ss, relation, ',') &&
            std::getline(ss, weight_str, ',')) {

            source = trim(source);
            target = trim(target);
            relation = trim(relation);

            if (source.empty() || target.empty() || relation.empty()) {
                continue;
            }

            float weight = 1.0f;
            try {
                weight = std::stof(weight_str);
            } catch (...) {
                weight = 1.0f;
            }

            json edge = json::object();
            edge["source"] = source;
            edge["target"] = target;
            edge["relation"] = relation;
            edge["weight"] = weight;
            edges_array.push_back(edge);
            edge_count++;
        }
    }

    csv_file.close();
    std::cerr << "[API] ✓ Loaded " << edge_count << " edges from CSV\n";
    std::cerr.flush();
    return edges_array;
}

std::string handleGraphProcessing(const Request& req) {
    json response;

    try {
        std::cerr << "\n[API] >>> POST /api/process\n";
        std::cerr << "[API] Body size: " << req.body.size() << " bytes\n";
        std::cerr.flush();

        if (req.body.size() > Config::MAX_BODY_SIZE) {
            response["status"] = "error";
            response["message"] = "Request size exceeded";
            std::cerr << "[API] ✗ Body size exceeded\n";
            std::cerr.flush();
            return response.dump();
        }

        std::string body_clean = trim(req.body);

        if (body_clean.empty()) {
            response["status"] = "error";
            response["message"] = "Empty request body";
            std::cerr << "[API] ✗ Empty body\n";
            std::cerr.flush();
            return response.dump();
        }

        json request_data;
        try {
            request_data = json::parse(body_clean);
            std::cerr << "[API] ✓ JSON parsed\n";
            std::cerr.flush();
        } catch (const json::exception& e) {
            response["status"] = "error";
            response["message"] = "Invalid JSON";
            std::cerr << "[API] ✗ JSON parse error: " << e.what() << "\n";
            std::cerr.flush();
            return response.dump();
        }

        if (!request_data.contains("text")) {
            response["status"] = "error";
            response["message"] = "Missing 'text' field";
            return response.dump();
        }

        std::string graph_type_field = "graph_type";
        if (!request_data.contains("graph_type") && !request_data.contains("graphType")) {
            response["status"] = "error";
            response["message"] = "Missing 'graph_type' field";
            return response.dump();
        }

        if (request_data.contains("graphType")) {
            graph_type_field = "graphType";
        }

        std::string text = request_data["text"].get<std::string>();
        std::string graph_type = request_data[graph_type_field].get<std::string>();

        std::cerr << "[API] Text length: " << text.size() << "\n";
        std::cerr << "[API] Graph type: " << graph_type << "\n";
        std::cerr.flush();

        json text_val = validateText(text);
        if (text_val["status"] != "success") {
            return text_val.dump();
        }

        json type_val = validateGraphType(graph_type);
        if (type_val["status"] != "success") {
            return type_val.dump();
        }

        std::cerr << "[API] ✓ Validations passed\n";
        std::cerr << "[API] Starting graph processing...\n";
        std::cerr.flush();

        std::string error_msg;
        json graph_data;
        std::string csv_file, html_file;
        bool success = false;

        try {
            success = runGraphProcessing(text, graph_type, error_msg, graph_data, csv_file, html_file);
        } catch (const std::exception& e) {
            std::cerr << "[API] ✗ Exception in runGraphProcessing: " << e.what() << "\n";
            std::cerr.flush();
            response["status"] = "error";
            response["message"] = "Processing error";
            response["error"] = std::string(e.what());
            return response.dump();
        }

        if (!success) {
            std::cerr << "[API] ✗ Processing failed: " << error_msg << "\n";
            std::cerr.flush();
            response["status"] = "error";
            response["message"] = error_msg;
            return response.dump();
        }

        std::cerr << "[API] ✓ Success! Graph processing completed\n";
        std::cerr.flush();

        json edges = loadEdgesFromCSV(csv_file);

        int node_count = 0;
        int edge_count = edges.size();

        if (edges.is_array() && edges.size() > 0) {
            std::set<std::string> unique_nodes;
            for (const auto& edge : edges) {
                if (edge.contains("source") && edge["source"].is_string()) {
                    unique_nodes.insert(edge["source"].get<std::string>());
                }
                if (edge.contains("target") && edge["target"].is_string()) {
                    unique_nodes.insert(edge["target"].get<std::string>());
                }
            }
            node_count = unique_nodes.size();
        }

        std::cerr << "[API] Graph info (for logs only): Nodes: " << node_count << ", Edges: " << edge_count << "\n";
        std::cerr.flush();

        response["status"] = "success";
        response["graph_type"] = graph_type;
        response["csv_file"] = csv_file;
        response["html_file"] = html_file;
        response["edges"] = edges;

        return response.dump();

    } catch (const std::exception& e) {
        std::cerr << "[API] ✗ FATAL: " << e.what() << "\n";
        std::cerr.flush();
        response["status"] = "error";
        response["message"] = "Server error";
        response["error"] = std::string(e.what());
        return response.dump();
    }
}

std::string handleSearchAnswer(const Request& req) {
    json response;

    try {
        std::cerr << "[API] >>> POST /api/search\n";
        std::cerr.flush();

        std::string body_clean = trim(req.body);

        if (body_clean.empty()) {
            response["status"] = "error";
            response["message"] = "Empty body";
            std::cerr << "[API] ✗ Empty body\n";
            std::cerr.flush();
            return response.dump();
        }

        std::cerr << "[API] Received body (first 200 chars): " << body_clean.substr(0, 200) << "\n";
        std::cerr.flush();

        json request_data = json::parse(body_clean);

        json graph_obj;
        if (request_data.contains("graph") && !request_data["graph"].is_null()) {
            std::cerr << "[API] ✓ Graph provided in request\n";
            std::cerr.flush();
            graph_obj = request_data["graph"];
        } else {
            std::cerr << "[API] ⚠️ Graph not in request, trying to load from file...\n";
            std::cerr.flush();
            std::string csv_path = "./temp/last_graph.csv";
            std::ifstream csv_file(csv_path);
            if (csv_file.is_open()) {
                graph_obj = json::object();
                graph_obj["edges"] = loadEdgesFromCSV(csv_path);
                csv_file.close();
                std::cerr << "[API] ✓ Loaded graph from file with " << graph_obj["edges"].size() << " edges\n";
                std::cerr.flush();
            } else {
                std::cerr << "[API] ⚠️ Could not load graph from file\n";
                std::cerr.flush();
                response["status"] = "warning";
                response["message"] = "Graph not provided and no cached graph found";
                response["answers"] = json::array();
                return response.dump();
            }
        }

        if (!request_data.contains("question") || request_data["question"].is_null()) {
            response["status"] = "error";
            response["message"] = "Missing required field: 'question'";
            std::cerr << "[API] ✗ Missing question field\n";
            std::cerr.flush();
            return response.dump();
        }

        std::string question = request_data["question"].get<std::string>();

        if (question.empty()) {
            response["status"] = "error";
            response["message"] = "Question cannot be empty";
            std::cerr << "[API] ✗ Empty question\n";
            std::cerr.flush();
            return response.dump();
        }

        std::cerr << "[API] ✓ Question: " << question << "\n";
        std::cerr << "[API] ✓ Graph edges: " << graph_obj["edges"].size() << "\n";
        std::cerr.flush();

        json search_payload = json::object();
        search_payload["question"] = question;
        search_payload["graph"] = graph_obj;

        std::string payload_str = search_payload.dump();

        std::cerr << "[API] Sending to Python search server...\n";
        std::cerr << "[API] Payload size: " << payload_str.size() << " bytes\n";
        std::cerr.flush();

        std::string temp_json = std::string(Config::TEMP_DIR) + "search_query.json";

        try {
            std::filesystem::create_directories(std::filesystem::path(Config::TEMP_DIR));
        } catch (...) {
        }


        std::ofstream temp_file(temp_json);
        if (!temp_file.is_open()) {
            std::cerr << "[API] ✗ Could not open temp file: " << temp_json << "\n";
            std::cerr.flush();
            response["status"] = "error";
            response["message"] = "Could not write search query";
            response["answers"] = json::array();
            return response.dump();
        }

        temp_file << payload_str;
        temp_file.close();

        std::ifstream check_file(temp_json);
        if (!check_file.is_open()) {
            std::cerr << "[API] ✗ Temp file not accessible after write: " << temp_json << "\n";
            std::cerr.flush();
            response["status"] = "error";
            response["message"] = "Temp file access error";
            response["answers"] = json::array();
            return response.dump();
        }
        check_file.close();

        std::cerr << "[API] ✓ Temp file created: " << temp_json << "\n";
        std::cerr.flush();
        std::string cmd;
        std::string python_executable;
        std::string python_script;

#ifdef _WIN32
        std::vector<std::string> py_candidates = {
            ".venv\\Scripts\\python.exe",
            "..\\.venv\\Scripts\\python.exe",
            "..\\..\\.venv\\Scripts\\python.exe"
        };

        std::vector<std::string> script_candidates = {
            std::string(Config::PYTHON_SCRIPT_DIR) + "search_kg_server.py", 
            "..\\python\\search_kg_server.py",
            "..\\..\\python\\search_kg_server.py"
        };

        for (const auto& p : py_candidates) {
            if (std::filesystem::exists(p)) { python_executable = p; break; }
        }

        for (const auto& s : script_candidates) {
            if (std::filesystem::exists(s)) { python_script = s; break; }
        }

        if (python_executable.empty() || python_script.empty()) {
            response["status"] = "error";
            response["message"] = "Python executable or script not found. Check .venv and python/ paths.";
            response["answers"] = json::array();
            return response.dump();
        }

        auto abs_py   = std::filesystem::absolute(python_executable).string();
        auto abs_scr  = std::filesystem::absolute(python_script).string();
        auto abs_json = std::filesystem::absolute(temp_json).string();

        auto norm = [](std::string s) {
            std::replace(s.begin(), s.end(), '\\', '/');
            return s;
        };
        abs_py   = norm(abs_py);
        abs_scr  = norm(abs_scr);
        abs_json = norm(abs_json);

        cmd = "cmd /C \"\"" + abs_py + "\" \"" + abs_scr + "\" \"" + abs_json + "\" 2>&1\"";

#else
        std::string python_executable = ".venv/bin/python3";
        std::string python_script = std::string(Config::PYTHON_SCRIPT_DIR) + "search_kg_server.py";

        auto abs_py   = std::filesystem::absolute(python_executable).string();
        auto abs_scr  = std::filesystem::absolute(python_script).string();
        auto abs_json = std::filesystem::absolute(temp_json).string();

        cmd = "\"" + abs_py + "\" \"" + abs_scr + "\" \"" + abs_json + "\" 2>&1";
#endif


        std::cerr << "[API] Debug - Python path: " << python_executable << "\n";
        std::cerr << "[API] Debug - Script path: " << python_script << "\n";
        std::cerr << "[API] Executing: " << cmd << "\n";
        std::cerr.flush();

        FILE* pipe_read = popen(cmd.c_str(), "r");
        if (!pipe_read) {
            std::cerr << "[API] ✗ Could not start Python search server\n";
            std::cerr.flush();
            response["status"] = "warning";
            response["message"] = "Search server unavailable";
            response["answers"] = json::array();
            return response.dump();
        }

        char buffer[65536];
        std::string python_result;
        while (fgets(buffer, sizeof(buffer), pipe_read) != NULL) {
            python_result += buffer;
        }

        int status_code = pclose(pipe_read);
        std::cerr << "[API] Python process exited with code: " << status_code << "\n";
        std::cerr << "[API] Response length: " << python_result.size() << " bytes\n";
        std::cerr.flush();

        if (python_result.empty()) {
            std::cerr << "[API] ⚠️ Python search returned empty result\n";
            std::cerr.flush();
            response["status"] = "warning";
            response["message"] = "⚠️ Search returned no answers";
            response["answers"] = json::array();
            return response.dump();
        }

        try {
            size_t json_start = python_result.find('{');
            if (json_start == std::string::npos) {
                std::cerr << "[API] ✗ No JSON found in Python response\n";
                std::cerr << "[API] Raw response: " << python_result.substr(0, 500) << "\n";
                std::cerr.flush();
                response["status"] = "warning";
                response["message"] = "Invalid Python response format";
                response["answers"] = json::array();
                return response.dump();
            }

            std::string json_str = python_result.substr(json_start);
            json py_response = json::parse(json_str);
            if (py_response.contains("error")) {
                response["status"] = "error";
                response["message"] = py_response["error"];
                response["answers"] = json::array();
                return response.dump();
            }


            std::cerr << "[API] ✓ Successfully parsed Python response\n";
            std::cerr.flush();

            if (py_response.contains("answers") && py_response["answers"].is_array()) {
                auto answers = py_response["answers"];
                std::cerr << "[API] ✓ Got " << answers.size() << " answers from Python\n";
                std::cerr.flush();

                response["status"] = "success";
                response["answers"] = answers;
                response["message"] = "✅ Search completed";
            } else {
                std::cerr << "[API] ⚠️ No 'answers' array in Python response\n";
                std::cerr << "[API] Python response keys: ";
                for (auto& el : py_response.items()) {
                    std::cerr << el.key() << " ";
                }
                std::cerr << "\n";
                std::cerr.flush();

                response["status"] = "warning";
                response["message"] = "No answers found";
                response["answers"] = json::array();
            }

        } catch (const json::exception& e) {
            std::cerr << "[API] ✗ Could not parse Python response: " << e.what() << "\n";
            std::cerr << "[API] Raw response (first 500 chars): " << python_result.substr(0, 500) << "\n";
            std::cerr.flush();

            response["status"] = "warning";
            response["message"] = "Search parsing error";
            response["answers"] = json::array();
        }

        return response.dump();

    } catch (const json::exception& e) {
        std::cerr << "[API] ✗ JSON Error: " << e.what() << "\n";
        std::cerr.flush();
        response["status"] = "error";
        response["message"] = "JSON parsing error: " + std::string(e.what());
        response["answers"] = json::array();
        return response.dump();

    } catch (const std::exception& e) {
        std::cerr << "[API] ✗ Exception: " << e.what() << "\n";
        std::cerr.flush();
        response["status"] = "error";
        response["message"] = std::string(e.what());
        response["answers"] = json::array();
        return response.dump();
    }
}

std::string handleGetGraphTypes(const Request& req) {
    (void)req;
    json response;
    response["status"] = "success";
    response["types"] = json::object({
        {"syntax", json::object({{"name", "Syntactic Graph"}})},
        {"semantic", json::object({{"name", "Semantic Graph"}})},
        {"hybrid", json::object({{"name", "Hybrid Graph"}})}
    });
    return response.dump();
}

std::string handleHealth(const Request& req) {
    (void)req;
    json response;
    response["status"] = "healthy";
    return response.dump();
}

std::string handleVersion(const Request& req) {
    (void)req;
    json response;
    response["version"] = Config::VERSION;
    return response.dump();
}