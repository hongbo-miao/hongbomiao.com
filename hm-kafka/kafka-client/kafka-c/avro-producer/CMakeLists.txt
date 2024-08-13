cmake_minimum_required(VERSION 3.28)
project(producer)

set(CMAKE_C_STANDARD 23)
set(CMAKE_C_FLAGS "-Wall -Werror -Wextra")
set(TARGET main)
file(GLOB SOURCE_FILES src/*.c)

find_package(PkgConfig REQUIRED)
pkg_check_modules(glib REQUIRED IMPORTED_TARGET glib-2.0)
pkg_check_modules(rdkafka REQUIRED IMPORTED_TARGET rdkafka)

include_directories(/opt/homebrew/opt/avro-c/include)
include_directories(/opt/homebrew/opt/libserdes/include)

link_directories(/opt/homebrew/opt/avro-c/lib)
link_directories(/opt/homebrew/opt/libserdes/lib)

add_executable(${TARGET} ${SOURCE_FILES})

target_link_libraries(${TARGET} avro serdes PkgConfig::glib PkgConfig::rdkafka)
