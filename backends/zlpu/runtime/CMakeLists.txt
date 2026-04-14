cmake_minimum_required(VERSION 3.19)
project(zlpu_runtime)

add_library(
  zlpu_backend STATIC ZLPUSBackend.cpp ZLPUDriverStub.cpp ZLPUMemoryManager.cpp
)

target_compile_options(zlpu_backend PRIVATE -std=c++17 -Wall -Wextra)

target_include_directories(
  zlpu_backend PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${EXECUTORCH_ROOT}/..
)
