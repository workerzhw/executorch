# Copyright (c) 2026 ZLPU Authors

cmake_minimum_required(VERSION 3.24)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

# Source root directory for executorch.
if(NOT EXECUTORCH_ROOT)
  set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../..)
endif()

include(${EXECUTORCH_ROOT}/tools/cmake/Utils.cmake)

add_library(zlpu_backend STATIC)
add_library(executorch::zlpu_backend ALIAS zlpu_backend)

target_sources(
  zlpu_backend
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/runtime/ZLPUSBackend.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/runtime/ZLPUDriverStub.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/runtime/ZLPUMemoryManager.cpp
)

target_include_directories(
  zlpu_backend
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${EXECUTORCH_ROOT} ${EXECUTORCH_ROOT}/..
          ${EXECUTORCH_ROOT}/runtime ${EXECUTORCH_ROOT}/runtime/core
          ${EXECUTORCH_ROOT}/runtime/core/portable_type
)

target_compile_options(zlpu_backend PRIVATE -Wall -Wextra)

target_link_libraries(zlpu_backend PRIVATE executorch)

install(
  TARGETS zlpu_backend
  EXPORT ExecuTorchTargets
  DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
