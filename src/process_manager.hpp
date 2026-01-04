#pragma once

#include <string>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

/**
 * @brief Запускает обработку графа с прямой передачей текста Python'у
 */
bool runGraphProcessing(
    const std::string& text,           // Текст в UTF-8
    const std::string& graph_type,     // "syntax", "semantic" или "hybrid"
    std::string& error_msg,            // Output: сообщение об ошибке если есть
    json& graph_data,                  // Output: данные графа в JSON
    std::string& csv_file,             // Output: путь к CSV файлу
    std::string& html_file             // Output: путь к HTML файлу
);