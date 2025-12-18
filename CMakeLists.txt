cmake_minimum_required(VERSION 3.20)
project(shadowtest)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find or fetch SDL3
find_package(SDL3 QUIET)
if(NOT SDL3_FOUND)
    include(FetchContent)
    FetchContent_Declare(
        SDL3
        GIT_REPOSITORY https://github.com/libsdl-org/SDL.git
        GIT_TAG main
    )
    FetchContent_MakeAvailable(SDL3)
endif()

# Fetch GLM for math
include(FetchContent)
FetchContent_Declare(
    glm
    GIT_REPOSITORY https://github.com/g-truc/glm.git
    GIT_TAG master
)
FetchContent_MakeAvailable(glm)

# OpenGL
find_package(OpenGL REQUIRED)

# Source files
add_executable(shadowtest
    shadowtest.cpp
    glad.c
)

target_include_directories(shadowtest PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${glm_SOURCE_DIR}
)

target_link_libraries(shadowtest PRIVATE
    OpenGL::GL
    SDL3::SDL3
)

# Platform-specific settings
if(WIN32)
    target_compile_definitions(shadowtest PRIVATE _CRT_SECURE_NO_WARNINGS)
endif()

