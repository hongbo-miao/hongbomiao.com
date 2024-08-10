cmake_minimum_required(VERSION 3.13)
project(producer)

set(CMAKE_C_STANDARD 99)
set(TARGET main)
set(SOURCE_FILES main.c)

find_package(PkgConfig REQUIRED)
pkg_check_modules(glib REQUIRED IMPORTED_TARGET glib-2.0)
pkg_check_modules(rdkafka REQUIRED IMPORTED_TARGET rdkafka)

include_directories(/opt/homebrew/opt/cjson/include)

link_directories(/opt/homebrew/opt/cjson/lib)

add_executable(${TARGET} ${SOURCE_FILES})

target_link_libraries(${TARGET} cjson PkgConfig::glib PkgConfig::rdkafka)
