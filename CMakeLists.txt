cmake_minimum_required(VERSION 3.11)
project(deform_transfer)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

### libIGL options: choose between header only and compiled static library
option(LIBIGL_USE_STATIC_LIBRARY     "Use libigl as static library" OFF)
option(LIBIGL_GLFW       "Use GLFW"           ON)

include(libigl)
igl_include(imgui)

# Find Eigen3 package
find_package(Eigen3 REQUIRED NO_MODULE)

# Add your project files
include_directories("include/")
file(GLOB SRCFILES src/*.cpp)
add_library(core ${SRCFILES}
        include/deform_transfer.h
        src/deform_transfer.cpp)
target_link_libraries(core igl::core igl::glfw )


add_executable(deform_transfer "main.cpp")
target_link_libraries(deform_transfer Eigen3::Eigen core igl::core igl::opengl igl::glfw igl::imgui)
