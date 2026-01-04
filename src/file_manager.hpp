#pragma once

#include <string>
#include <vector>

namespace FileManager {

/**
 * @brief Получить директорию для временных файлов
 * @return Путь к директории временных файлов
 */
std::string getTempDir();

/**
 * @brief Сохранить контент во временный файл
 * @param content Содержимое
 * @param ext Расширение файла (например ".txt")
 * @return Путь к созданному файлу
 */
std::string saveTempFile(const std::string& content, const std::string& ext);

/**
 * @brief Сохранить JSON конфигурацию
 * @param jsonStr JSON строка
 * @return Путь к конфигурационному файлу
 */
std::string saveConfig(const std::string& jsonStr);

/**
 * @brief Прочитать содержимое файла
 * @param path Путь к файлу
 * @return Содержимое файла
 * @throw std::runtime_error если файл не найден
 */
std::string readFile(const std::string& path);

/**
 * @brief Удалить файл
 * @param path Путь к файлу
 * @return true если успешно удалено
 */
bool deleteFile(const std::string& path);

/**
 * @brief Проверить существование файла
 * @param path Путь к файлу
 * @return true если файл существует
 */
bool fileExists(const std::string& path);

/**
 * @brief Создать директорию
 * @param path Путь к директории
 * @return true если успешно создана
 */
bool createDirectory(const std::string& path);

} 