
cmake_minimum_required(VERSION 3.15)
project(GroupActivityRecognition LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
// find_package(Caffe REQUIRED)          // uncomment when you install Caffe
// find_package(Torch REQUIRED)          // or LibTorch if preferred

include_directories(${OpenCV_INCLUDE_DIRS})

# Example: data preparation tool
add_executable(extract_crops src/data_prep/extract_crops.cpp)
target_link_libraries(extract_crops ${OpenCV_LIBS})

# Add more executables here...
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
