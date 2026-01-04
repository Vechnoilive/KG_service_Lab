#pragma once

#include <string>
#include <vector>
#include <map>
#include <functional>

struct Request {
    std::string method;
    std::string path;
    std::string body;
    std::map<std::string, std::string> headers;
};

typedef std::function<std::string(const Request&)> RequestHandler;

struct Endpoint {
    std::string method;
    std::string path;
    RequestHandler handler;
};

class HTTPServer {
private:
    std::vector<Endpoint> endpoints;
    std::map<std::string, std::string> staticDirs;

public:
    HTTPServer();
    ~HTTPServer();

    void registerEndpoint(const std::string& method, const std::string& path, RequestHandler handler);
    void serveStatic(const std::string& mount, const std::string& directory);
    void listen(int port, const std::string& host);

private:
    bool readStaticFile(const std::string& filepath, std::string& content);
    std::string getContentType(const std::string& path);
};

Request parseHTTPRequest(const std::string& raw);
std::string formatHTTPResponse(int statusCode, const std::string& statusMsg, const std::string& body, const std::string& contentType);