#pragma once

#include "server.hpp"
#include <nlohmann/json.hpp>
#include <string>

using json = nlohmann::json;

std::string handleGraphProcessing(const Request& req);
std::string handleGetGraphTypes(const Request& req);
std::string handleSearchAnswer(const Request& req);
std::string handleHealth(const Request& req);
std::string handleVersion(const Request& req);