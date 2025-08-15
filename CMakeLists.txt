cmake_minimum_required(VERSION 3.16)
#include <cstring>  // pentru strstr
# Set the project name
project(gpu_miner_opencl)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenCL
find_package(OpenCL REQUIRED)

# Find ncurses
find_package(Curses REQUIRED)

# Specify where to find the headers
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${OpenCL_INCLUDE_DIRS})
include_directories(${CURSES_INCLUDE_DIR})

# Define the source files
set(SRC_FILES
    src/main.cpp
    src/opencl_utils.cpp
    src/sha256.cpp
)

# Create the executable
add_executable(gpu_miner_opencl ${SRC_FILES})

# Link libraries
target_link_libraries(gpu_miner_opencl 
    ${OpenCL_LIBRARIES} 
    ${CURSES_LIBRARIES}
    pthread
    rt
)

# Optional: Add compile options
target_compile_options(gpu_miner_opencl PRIVATE -Wall -O3)