# Minimum CMake version required
cmake_minimum_required(VERSION 3.10)

# Project name and version
project(ConeDetection VERSION 1.0)

# Need OpenCV
# find_package(OpenCV REQUIRED) # won't work for displaying images

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Add source files
file(GLOB SOURCES ${PROJECT_SOURCE_DIR}/code/main.cpp
                    ${PROJECT_SOURCE_DIR}/code/ConeFilter.cpp
                    ${PROJECT_SOURCE_DIR}/code/DBSCAN.cpp)

message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV libs: ${OpenCV_LIBS}")

# Add the executable
add_executable(ConeDetection ${SOURCES})

# Link the executable to the libraries in the lib directory
target_include_directories(ConeDetection PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(ConeDetection PUBLIC ${OpenCV_LIBS})

#file(COPY "${PROJECT_SOURCE_DIR}/graphics" DESTINATION "${COMMON_OUTPUT_DIR}/bin/")
#file(COPY "${PROJECT_SOURCE_DIR}/fonts" DESTINATION "${COMMON_OUTPUT_DIR}/bin/")
#file(COPY "${PROJECT_SOURCE_DIR}/low res" DESTINATION "${COMMON_OUTPUT_DIR}/bin/")
#file(COPY "${PROJECT_SOURCE_DIR}/sound" DESTINATION "${COMMON_OUTPUT_DIR}/bin/")

# to run: ./ConeDetection ~/scratch/Autonomous/fsoco/fsoco/amz
