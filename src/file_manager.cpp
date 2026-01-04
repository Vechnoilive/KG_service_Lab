#include "file_manager.hpp"
#include "config.hpp"
#include <fstream>
#include <sstream>
#include <iostream>
#include <chrono>
#include <sys/stat.h>
#include <cstdlib>

#ifdef _WIN32
#include <direct.h>
#endif

namespace FileManager {

/**
 * @brief Получить директорию для временных файлов
 * @return Путь к директории временных файлов
 */
std::string getTempDir() {
    return Config::TEMP_DIR;
}

/**
 * @brief Сохранить контент во временный файл с UTF-8 BOM маркером
 * @param content Содержимое (UTF-8 строка)
 * @param ext Расширение файла (например ".txt")
 * @return Путь к созданному файлу
 */
std::string saveTempFile(const std::string& content, const std::string& ext) {
    if (content.length() > Config::MAX_TEXT_LENGTH) {
        throw std::runtime_error("Content exceeds maximum size: " +
            std::to_string(content.length()) + " > " +
            std::to_string(Config::MAX_TEXT_LENGTH));
    }

    createDirectory(Config::TEMP_DIR);

    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()
    ) % 1000;
    
    std::stringstream filename;
    filename << Config::TEMP_DIR << "input_" << time << "_" << ms.count() << ext;
    
    std::ofstream file(filename.str(), std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot create temp file: " + filename.str());
    }
    
    const unsigned char utf8_bom[] = { 0xEF, 0xBB, 0xBF };
    file.write((const char*)utf8_bom, 3);
    
    file.write(content.c_str(), content.length());
    file.close();
    
    std::cout << " ├─ Temp file: " << filename.str()
        << " (" << content.length() << " bytes)" << std::endl;
    
    return filename.str();
}

/**
 * @brief Сохранить JSON конфигурацию с UTF-8 BOM маркером
 * @param jsonStr JSON строка
 * @return Путь к конфигурационному файлу
 */
std::string saveConfig(const std::string& jsonStr) {
    createDirectory(Config::TEMP_DIR);

    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    
    std::stringstream filename;
    filename << Config::TEMP_DIR << "config_" << time << ".json";
    
    std::ofstream file(filename.str(), std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot create config file: " + filename.str());
    }
    
    const unsigned char utf8_bom[] = { 0xEF, 0xBB, 0xBF };
    file.write((const char*)utf8_bom, 3);
    file.write(jsonStr.c_str(), jsonStr.length());
    file.close();
    
    std::cout << " ├─ Config file: " << filename.str() << std::endl;
    
    return filename.str();
}

/**
 * @brief Прочитать содержимое файла в бинарном режиме и пропустить BOM маркер если есть
 * @param path Путь к файлу
 * @return Содержимое файла (БЕЗ BOM маркера если он был)
 * @throw std::runtime_error если файл не найден
 */
std::string readFile(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + path);
    }
    
    std::stringstream buffer;
    buffer << file.rdbuf();
    file.close();
    
    std::string content = buffer.str();

    if (content.length() >= 3 &&
        (unsigned char)content[0] == 0xEF &&
        (unsigned char)content[1] == 0xBB &&
        (unsigned char)content[2] == 0xBF) {
        std::cout << " ├─ Skipping UTF-8 BOM marker" << std::endl;
        content = content.substr(3);
    }
    
    return content;
}

/**
 * @brief Удалить файл
 * @param path Путь к файлу
 * @return true если успешно удалено
 */
bool deleteFile(const std::string& path) {
    return std::remove(path.c_str()) == 0;
}

/**
 * @brief Проверить существование файла
 * @param path Путь к файлу
 * @return true если файл существует
 */
bool fileExists(const std::string& path) {
    struct stat buffer;
    return (stat(path.c_str(), &buffer) == 0);
}

/**
 * @brief Создать директорию
 * @param path Путь к директории
 * @return true если успешно создана
 */
bool createDirectory(const std::string& path) {
#ifdef _WIN32
    return _mkdir(path.c_str()) == 0 || fileExists(path);
#else
    return mkdir(path.c_str(), 0755) == 0 || fileExists(path);
#endif
}

}