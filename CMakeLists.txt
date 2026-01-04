cmake_minimum_required(VERSION 3.10)
project(graph_server)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ============ FIND OR FETCH nlohmann/json ============
message(STATUS "Finding nlohmann_json...")

# Попытка 1: Поиск через find_package
find_package(nlohmann_json QUIET)

if(nlohmann_json_FOUND)
    message(STATUS "✓ Found nlohmann_json via find_package")
else()
    message(STATUS "⚠️ nlohmann_json not found via find_package, checking local include...")
    
    # Попытка 2: Проверка локального файла
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/include/nlohmann/json.hpp")
        message(STATUS "✓ Found local nlohmann/json.hpp")
    else()
        message(STATUS "⚠️ Local file not found, will try FetchContent...")
        
        # Попытка 3: Автоматическое скачивание
        include(FetchContent)
        FetchContent_Declare(json 
            URL https://github.com/nlohmann/json/releases/download/v3.11.2/json.hpp
        )
        FetchContent_MakeAvailable(json)
    endif()
endif()

# ============ SOURCE FILES ============
set(SOURCES
    src/main.cpp
    src/server.cpp
    src/api_handler.cpp
    src/process_manager.cpp
    src/file_manager.cpp
)

set(HEADERS
    src/server.hpp
    src/api_handler.hpp
    src/process_manager.hpp
    src/file_manager.hpp
    src/config.hpp
)

# ============ EXECUTABLE ============
add_executable(graph_server ${SOURCES} ${HEADERS})

# ============ INCLUDE DIRECTORIES ============
target_include_directories(graph_server PRIVATE 
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
)

# ============ LINK nlohmann_json ============
if(TARGET nlohmann_json::nlohmann_json)
    target_link_libraries(graph_server PRIVATE nlohmann_json::nlohmann_json)
endif()

# ============ WINDOWS SPECIFIC ============
if(WIN32)
    message(STATUS "Windows detected")
    if(MINGW)
        target_link_libraries(graph_server PRIVATE ws2_32)
        message(STATUS "✓ Linked Windows libraries (ws2_32) for MinGW")
    else()
        target_link_libraries(graph_server PRIVATE ws2_32 wsock32)
        message(STATUS "✓ Linked Windows libraries (ws2_32, wsock32) for MSVC")
    endif()
else()
    message(STATUS "Unix/Linux detected")
endif()

# ============ BUILD CONFIGURATION ============
message(STATUS "")
message(STATUS "===== BUILD CONFIGURATION =====")
message(STATUS "Project: ${PROJECT_NAME}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "================================")