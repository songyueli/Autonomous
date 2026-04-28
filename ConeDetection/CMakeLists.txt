# Minimum CMake version required
cmake_minimum_required(VERSION 3.18)
 
# Project name and version
project(ConeDetection VERSION 1.0 LANGUAGES CXX)
 
# Options for different build types
option(USE_OPENMP "Enable OpenMP parallelization" ON)
option(USE_CUDA "Enable CUDA GPU acceleration" ON)
 
# Specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
 
# Optimization flags
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native")
 
# Find OpenMP
find_package(OpenMP REQUIRED)
message(STATUS "OpenMP found: ${OpenMP_CXX_VERSION}")

# Enable CUDA if requested
if(USE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    
    set(CMAKE_CUDA_ARCHITECTURES 75)
    
    message(STATUS "CUDA found: ${CUDAToolkit_VERSION}")
    message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
endif()
 
# Common source files
set(COMMON_SOURCES
    ${PROJECT_SOURCE_DIR}/code/ConeFilter.cpp
    ${PROJECT_SOURCE_DIR}/code/DBSCAN.cpp
)
 
# CUDA sources (only if CUDA is enabled)
if(USE_CUDA)
    set(CUDA_SOURCES
        ${PROJECT_SOURCE_DIR}/code/ConeFilterCUDA.cu
    )
    set_source_files_properties(${CUDA_SOURCES} PROPERTIES LANGUAGE CUDA)
endif()
 
# Determine which main file to use
if(USE_OPENMP AND USE_CUDA)
    set(MAIN_SOURCE ${PROJECT_SOURCE_DIR}/code/main_hybrid.cpp)
    set(EXECUTABLE_NAME ConeDetection_hybrid)
    message(STATUS "Building HYBRID version (OpenMP + CUDA)")
elseif(USE_CUDA)
    set(MAIN_SOURCE ${PROJECT_SOURCE_DIR}/code/main.cpp)
    set(EXECUTABLE_NAME ConeDetection_cuda)
    message(STATUS "Building CUDA version")
elseif(USE_OPENMP)
    set(MAIN_SOURCE ${PROJECT_SOURCE_DIR}/code/main_openmp.cpp)
    set(EXECUTABLE_NAME ConeDetection_openmp)
    message(STATUS "Building OpenMP version")
else()
    set(MAIN_SOURCE ${PROJECT_SOURCE_DIR}/code/main.cpp)
    set(EXECUTABLE_NAME ConeDetection)
    message(STATUS "Building ORIGINAL version (no parallelization)")
endif()
 
# Create executable
if(USE_CUDA)
    add_executable(${EXECUTABLE_NAME} ${MAIN_SOURCE} ${COMMON_SOURCES} ${CUDA_SOURCES})
    
    # Set CUDA properties
    set_target_properties(${EXECUTABLE_NAME} PROPERTIES 
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_STANDARD 14
    )
    
    target_link_libraries(${EXECUTABLE_NAME} PRIVATE CUDA::cudart)
else()
    add_executable(${EXECUTABLE_NAME} ${MAIN_SOURCE} ${COMMON_SOURCES})
endif()
 
# Link OpenMP if enabled
target_link_libraries(${EXECUTABLE_NAME} PRIVATE OpenMP::OpenMP_CXX)
 
# Include directories
target_include_directories(${EXECUTABLE_NAME} PRIVATE ${PROJECT_SOURCE_DIR}/code)
 
# Link filesystem library if needed
target_link_libraries(${EXECUTABLE_NAME} PRIVATE stdc++fs)
 
# Print build configuration
message(STATUS "")
message(STATUS "=== Build Configuration ===")
message(STATUS "Executable: ${EXECUTABLE_NAME}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "OpenMP: ${USE_OPENMP}")
message(STATUS "CUDA: ${USE_CUDA}")
if(USE_CUDA)
    message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
endif()
message(STATUS "===========================")
message(STATUS "")
 
# Installation
install(TARGETS ${EXECUTABLE_NAME} DESTINATION bin)

add_custom_target(build-all
    COMMAND ${CMAKE_COMMAND} -S ${PROJECT_SOURCE_DIR} -B build_cpu -DUSE_OPENMP=OFF -DUSE_CUDA=OFF
    COMMAND ${CMAKE_COMMAND} --build build_cpu -j24

    COMMAND ${CMAKE_COMMAND} -S ${PROJECT_SOURCE_DIR} -B build_openmp -DUSE_OPENMP=ON -DUSE_CUDA=OFF
    COMMAND ${CMAKE_COMMAND} --build build_openmp -j24

    COMMAND ${CMAKE_COMMAND} -S ${PROJECT_SOURCE_DIR} -B build_hybrid -DUSE_OPENMP=ON -DUSE_CUDA=ON
    COMMAND ${CMAKE_COMMAND} --build build_hybrid -j24

    COMMENT "Building all versions (CPU, OpenMP, CUDA, Hybrid)"
)