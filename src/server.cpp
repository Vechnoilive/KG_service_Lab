#include "server.hpp"
#include "api_handler.hpp"
#include "file_manager.hpp"
#include "config.hpp"
#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <vector>
#include <map>

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#else
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#endif

HTTPServer::HTTPServer() {}

HTTPServer::~HTTPServer() {}

void HTTPServer::registerEndpoint(const std::string& method, const std::string& path, RequestHandler handler) {
    endpoints.push_back({method, path, handler});
    std::cout << "[SERVER] Registered endpoint: " << method << " " << path << std::endl;
}

void HTTPServer::serveStatic(const std::string& mount, const std::string& directory) {
    staticDirs[mount] = directory;
    std::cout << "[SERVER] Mounting static directory: " << mount << " -> " << directory << std::endl;
}

std::string getFileContentImpl(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        return "";
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

std::string getContentTypeImpl(const std::string& path) {
    if (path.find(".html") != std::string::npos) return "text/html";
    if (path.find(".css") != std::string::npos) return "text/css";
    if (path.find(".js") != std::string::npos) return "application/javascript";
    if (path.find(".json") != std::string::npos) return "application/json";
    if (path.find(".csv") != std::string::npos) return "text/csv";
    if (path.find(".png") != std::string::npos) return "image/png";
    if (path.find(".jpg") != std::string::npos || path.find(".jpeg") != std::string::npos) return "image/jpeg";
    return "application/octet-stream";
}

std::string buildResponseImpl(int status_code, const std::string& body, const std::string& content_type = "application/json") {
    std::string status_text;
    switch (status_code) {
    case 200: status_text = "OK"; break;
    case 201: status_text = "Created"; break;
    case 400: status_text = "Bad Request"; break;
    case 404: status_text = "Not Found"; break;
    case 500: status_text = "Internal Server Error"; break;
    default: status_text = "Unknown"; break;
    }

    std::ostringstream response;
    response << "HTTP/1.1 " << status_code << " " << status_text << "\r\n";
    response << "Content-Type: " << content_type << "; charset=utf-8\r\n";
    response << "Content-Length: " << body.length() << "\r\n";
    response << "Access-Control-Allow-Origin: *\r\n";
    response << "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n";
    response << "Access-Control-Allow-Headers: Content-Type\r\n";
    response << "Connection: close\r\n";
    response << "\r\n";
    response << body;
    return response.str();
}

Request parseRequestImpl(const std::string& raw) {
    Request req;

    size_t header_end = raw.find("\r\n\r\n");
    size_t body_start = 0;

    if (header_end != std::string::npos) {
        body_start = header_end + 4;
    } else {
        header_end = raw.find("\n\n");
        if (header_end != std::string::npos) {
            body_start = header_end + 2;
        }
    }

    std::string headers_part = raw.substr(0, header_end != std::string::npos ? header_end : raw.find('\n'));
    std::istringstream iss(headers_part);
    std::string line;

    if (std::getline(iss, line)) {
        std::istringstream method_line(line);
        std::string version;
        method_line >> req.method >> req.path >> version;

        size_t query_pos = req.path.find('?');
        if (query_pos != std::string::npos) {
            req.path = req.path.substr(0, query_pos);
        }
    }

    int content_length = 0;
    while (std::getline(iss, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }

        if (line.empty()) {
            break;
        }

        size_t colon_pos = line.find(':');
        if (colon_pos != std::string::npos) {
            std::string key = line.substr(0, colon_pos);
            std::string value = line.substr(colon_pos + 2);
            req.headers[key] = value;

            if (key == "Content-Length") {
                try {
                    content_length = std::stoi(value);
                } catch (...) {
                    content_length = 0;
                }
            }
        }
    }

    if (body_start < raw.size()) {
        int available = raw.size() - body_start;
        int to_read = (content_length > 0 && content_length <= available) ? content_length : available;
        if (to_read > 0) {
            req.body = raw.substr(body_start, to_read);
        }
    }

    std::cerr << "[PARSE] Content-Length: " << content_length << ", Body size: " << req.body.size() << std::endl;

    return req;
}

void HTTPServer::listen(int port, const std::string& host) {
    (void)host;

#ifdef _WIN32
    WSADATA wsa_data;
    if (WSAStartup(MAKEWORD(2, 2), &wsa_data) != 0) {
        std::cerr << "[SERVER] WSAStartup failed" << std::endl;
        return;
    }

    SOCKET server_socket = socket(AF_INET, SOCK_STREAM, 0);
    if (server_socket == INVALID_SOCKET) {
        std::cerr << "[SERVER] Error creating socket" << std::endl;
        WSACleanup();
        return;
    }

    const char opt = 1;
    if (setsockopt(server_socket, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)) < 0) {
        std::cerr << "[SERVER] Error setting socket options" << std::endl;
        closesocket(server_socket);
        WSACleanup();
        return;
    }

    struct sockaddr_in server_addr;
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(port);
    server_addr.sin_addr.s_addr = INADDR_ANY;

    if (bind(server_socket, (struct sockaddr *)&server_addr, sizeof(server_addr)) < 0) {
        std::cerr << "[SERVER] Error binding socket" << std::endl;
        closesocket(server_socket);
        WSACleanup();
        return;
    }

    ::listen(server_socket, 5);

    std::cout << "\n========================================" << std::endl;
    std::cout << "Server started on port " << port << std::endl;
    std::cout << "Knowledge Graph Builder API v" << Config::VERSION << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Registered endpoints:" << std::endl;
    for (const auto& ep : endpoints) {
        std::cout << " " << ep.method << " " << ep.path << std::endl;
    }
    std::cout << "Static directories:" << std::endl;
    for (const auto& [mount, dir] : staticDirs) {
        std::cout << " " << mount << " -> " << dir << std::endl;
    }
    std::cout << "========================================\n" << std::endl;

    while (true) {
        struct sockaddr_in client_addr;
        int client_addr_len = sizeof(client_addr);

        SOCKET client_socket = accept(server_socket, (struct sockaddr *)&client_addr, &client_addr_len);
        if (client_socket == INVALID_SOCKET) {
            std::cerr << "[SERVER] Error accepting connection" << std::endl;
            continue;
        }

        std::string raw_request;
        char buffer[8192];
        int total_received = 0;
        int max_size = 1024 * 1024;

        while (total_received < max_size) {
            int bytes_received = recv(client_socket, buffer, sizeof(buffer), 0);

            if (bytes_received > 0) {
                raw_request.append(buffer, bytes_received);
                total_received += bytes_received;

                size_t header_end = raw_request.find("\r\n\r\n");
                if (header_end != std::string::npos) {
                    size_t cl_pos = raw_request.find("Content-Length:");
                    if (cl_pos != std::string::npos) {
                        size_t cl_end = raw_request.find("\r\n", cl_pos);
                        std::string cl_str = raw_request.substr(cl_pos + 15, cl_end - (cl_pos + 15));
                        try {
                            int content_length = std::stoi(cl_str);
                            int body_start = header_end + 4;
                            int received_body = raw_request.size() - body_start;

                            if (received_body >= content_length) {
                                break;
                            }
                        } catch (...) {
                            break;
                        }
                    } else {
                        break;
                    }
                }
            } else if (bytes_received == 0) {
                break;
            } else {
                break;
            }
        }

        if (!raw_request.empty()) {
            Request req = parseRequestImpl(raw_request);
            std::cout << "[SERVER] " << req.method << " " << req.path << std::endl;

            std::string response;
            bool handled = false;

            for (const auto& ep : endpoints) {
                if (ep.method == req.method && ep.path == req.path) {
                    std::cout << "[SERVER] ✓ Handled by endpoint: " << ep.path << std::endl;
                    response = buildResponseImpl(200, ep.handler(req), "application/json");
                    handled = true;
                    break;
                }
            }

            if (!handled) {
                if (req.path == "/" || req.path == "/index.html") {
                    std::string index_path = "./web/index.html";
                    std::cout << "[SERVER] Looking for index.html at: " << index_path << std::endl;
                    std::string content = getFileContentImpl(index_path);

                    if (!content.empty()) {
                        std::cout << "[SERVER] ✓ Served index.html (" << content.size() << " bytes)" << std::endl;
                        std::ostringstream html_response;
                        html_response << "HTTP/1.1 200 OK\r\n";
                        html_response << "Content-Type: text/html; charset=utf-8\r\n";
                        html_response << "Content-Length: " << content.length() << "\r\n";
                        html_response << "Cache-Control: no-cache, no-store, must-revalidate, max-age=0, private\r\n";
                        html_response << "Pragma: no-cache\r\n";
                        html_response << "Expires: -1\r\n";
                        html_response << "Vary: Accept-Encoding\r\n";
                        html_response << "Access-Control-Allow-Origin: *\r\n";
                        html_response << "Connection: close\r\n";
                        html_response << "\r\n";
                        html_response << content;
                        response = html_response.str();
                        handled = true;
                    } else {
                        std::cout << "[SERVER] ✗ index.html not found at " << index_path << std::endl;
                    }
                }
            }

            if (!handled) {
                for (const auto& [mount, directory] : staticDirs) {
                    if (req.path.find(mount) == 0) {
                        std::string relative_path = req.path.substr(mount.length());
                        if (relative_path.empty()) {
                            relative_path = "index.html";
                        }

                        std::string file_path = directory + relative_path;
                        std::cout << "[SERVER] Looking for file at: " << file_path << std::endl;
                        std::string content = getFileContentImpl(file_path);

                        if (!content.empty()) {
                            std::cout << "[SERVER] ✓ Served: " << file_path << std::endl;
                            std::string content_type = getContentTypeImpl(file_path);
                            response = buildResponseImpl(200, content, content_type);
                            handled = true;
                            break;
                        }
                    }
                }
            }

            if (!handled) {
                std::cout << "[SERVER] ✗ Not found: " << req.path << std::endl;
                response = buildResponseImpl(404, "{\"error\": \"Not found\"}", "application/json");
            }

            send(client_socket, response.c_str(), (int)response.length(), 0);
        }

        closesocket(client_socket);
    }

    closesocket(server_socket);
    WSACleanup();

#else
    // Unix/Linux version
    int server_socket = socket(AF_INET, SOCK_STREAM, 0);
    if (server_socket < 0) {
        std::cerr << "[SERVER] Error creating socket" << std::endl;
        return;
    }

    int opt = 1;
    if (setsockopt(server_socket, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)) < 0) {
        std::cerr << "[SERVER] Error setting socket options" << std::endl;
        close(server_socket);
        return;
    }

    struct sockaddr_in server_addr;
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(port);
    server_addr.sin_addr.s_addr = INADDR_ANY;

    if (bind(server_socket, (struct sockaddr *)&server_addr, sizeof(server_addr)) < 0) {
        std::cerr << "[SERVER] Error binding socket" << std::endl;
        close(server_socket);
        return;
    }

    ::listen(server_socket, 5);

    std::cout << "\n========================================" << std::endl;
    std::cout << "🚀 Server started on port " << port << std::endl;
    std::cout << "📊 Knowledge Graph Builder API v" << Config::VERSION << std::endl;
    std::cout << "========================================" << std::endl;

    while (true) {
        struct sockaddr_in client_addr;
        socklen_t client_addr_len = sizeof(client_addr);

        int client_socket = accept(server_socket, (struct sockaddr *)&client_addr, &client_addr_len);
        if (client_socket < 0) {
            std::cerr << "[SERVER] Error accepting connection" << std::endl;
            continue;
        }

        std::string raw_request;
        char buffer[8192];
        int total_received = 0;

        while (total_received < 1024 * 1024) {
            int bytes_received = recv(client_socket, buffer, sizeof(buffer), 0);

            if (bytes_received > 0) {
                raw_request.append(buffer, bytes_received);
                total_received += bytes_received;

                size_t header_end = raw_request.find("\r\n\r\n");
                if (header_end != std::string::npos) {
                    size_t cl_pos = raw_request.find("Content-Length:");
                    if (cl_pos != std::string::npos) {
                        try {
                            int content_length = std::stoi(raw_request.substr(cl_pos + 15));
                            int body_start = header_end + 4;
                            if ((int)raw_request.size() - body_start >= content_length) {
                                break;
                            }
                        } catch (...) {
                            break;
                        }
                    } else {
                        break;
                    }
                }
            } else {
                break;
            }
        }

        if (!raw_request.empty()) {
            Request req = parseRequestImpl(raw_request);
            std::cout << "[SERVER] " << req.method << " " << req.path << std::endl;

            std::string response;
            bool handled = false;

            for (const auto& ep : endpoints) {
                if (ep.method == req.method && ep.path == req.path) {
                    std::cout << "[SERVER] ✓ Handled by endpoint: " << ep.path << std::endl;
                    response = buildResponseImpl(200, ep.handler(req), "application/json");
                    handled = true;
                    break;
                }
            }

            if (!handled) {
                if (req.path == "/" || req.path == "/index.html") {
                    std::string content = getFileContentImpl("./web/index.html");
                    if (!content.empty()) {
                        response = buildResponseImpl(200, content, "text/html");
                        handled = true;
                    }
                }
            }

            if (!handled) {
                response = buildResponseImpl(404, "{\"error\": \"Not found\"}", "application/json");
            }

            send(client_socket, response.c_str(), response.length(), 0);
        }

        close(client_socket);
    }

    close(server_socket);

#endif
}

Request parseHTTPRequest(const std::string& raw) {
    return parseRequestImpl(raw);
}

std::string formatHTTPResponse(int statusCode, const std::string& statusMsg, const std::string& body, const std::string& contentType) {
    std::ostringstream response;
    response << "HTTP/1.1 " << statusCode << " " << statusMsg << "\r\n";
    response << "Content-Type: " << contentType << "; charset=utf-8\r\n";
    response << "Content-Length: " << body.length() << "\r\n";
    response << "Connection: close\r\n";
    response << "\r\n";
    response << body;
    return response.str();
}