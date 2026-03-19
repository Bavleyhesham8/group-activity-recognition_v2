// create_project_structure.cpp
// Creates organized folder structure for Group Activity Recognition C++ project
// Compile with: g++ -std=c++17 create_project_structure.cpp -o create_struct
// Run: ./create_struct

#include <iostream>
#include <filesystem>
#include <vector>
#include <string>
#include <fstream>              // ← Added this to fix ofstream error

namespace fs = std::filesystem;

void create_directory(const std::string& path) {
    try {
        if (fs::create_directories(path)) {
            std::cout << "Created directory: " << path << "\n";
        } else {
            std::cout << "Directory already exists: " << path << "\n";
        }
    } catch (const fs::filesystem_error& e) {
        std::cerr << "Error creating " << path << ": " << e.what() << "\n";
    }
}

void create_file(const std::string& path, const std::string& content = "") {
    if (fs::exists(path)) {
        std::cout << "File already exists (skipped): " << path << "\n";
        return;
    }

    std::ofstream file(path);
    if (!file.is_open()) {
        std::cerr << "Failed to create file: " << path << "\n";
        return;
    }

    if (!content.empty()) {
        file << content;
    }
    file.close();
    std::cout << "Created file: " << path << "\n";
}

int main() {
    std::string root = "group-activity-recognition-cpp";

    std::cout << "Creating project structure in: ./" << root << "\n\n";

    // Root folders
    std::vector<std::string> directories = {
        root,
        root + "/data/raw/volleyball",
        root + "/data/processed/frames",
        root + "/data/processed/crops",
        root + "/data/processed/features",
        root + "/data/processed/sequences",
        root + "/src/data_prep",
        root + "/src/models",
        root + "/src/utils",
        root + "/configs",
        root + "/outputs/checkpoints",
        root + "/outputs/logs",
        root + "/outputs/results",
        root + "/include",
        root + "/build",
        root + "/scripts"
    };

    for (const auto& dir : directories) {
        create_directory(dir);
    }

    // Placeholder CMake file (minimal skeleton)
    std::string cmake_content = R"(
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
)";

    create_file(root + "/CMakeLists.txt", cmake_content);

    // Empty source skeletons
    create_file(root + "/src/data_prep/extract_crops.cpp",
                "// extract_crops.cpp - crop players using annotations\n#include <opencv2/opencv.hpp>\nint main() { return 0; }\n");
    create_file(root + "/src/data_prep/extract_features.cpp",
                "// extract_features.cpp - AlexNet fc7 with Caffe or LibTorch\nint main() { return 0; }\n");
    create_file(root + "/src/data_prep/build_sequences.cpp",
                "// build_sequences.cpp - create 9-frame sequences\nint main() { return 0; }\n");

    create_file(root + "/src/train_stage1.cpp",
                "// train_stage1.cpp - Person-level LSTM training\nint main() { return 0; }\n");
    create_file(root + "/src/train_stage2.cpp",
                "// train_stage2.cpp - Group-level LSTM training\nint main() { return 0; }\n");

    // Config placeholder (JSON example)
    std::string config_content = R"({
  "timesteps": 9,
  "batch_size": 4,
  "learning_rate": 0.00001,
  "momentum": 0.9,
  "hidden_size_person": 3000,
  "hidden_size_group": 500,
  "dataset_root": "data/raw/volleyball"
})";
    create_file(root + "/configs/config.json", config_content);

    // README
    std::string readme_content = R"(# Group Activity Recognition (CVPR 2016 Paper Implementation)

C++ re-implementation of "A Hierarchical Deep Temporal Model for Group Activity Recognition"

## Structure
- data/          → raw dataset + processed features/sequences
- src/           → all source code
- configs/       → hyperparameters
- outputs/       → checkpoints, logs, confusion matrices

## Build
mkdir build && cd build
cmake ..
make

## Next steps
1. Place volleyball dataset in data/raw/volleyball/
2. Implement extract_crops.cpp using provided bounding boxes
3. Use Caffe or LibTorch for AlexNet + LSTM
)";
    create_file(root + "/README.md", readme_content);

    // Simple run script example
    std::string run_content = R"(#!/bin/bash
echo "Project setup complete."
echo "Next: cd build && cmake .. && make"
)";
    create_file(root + "/scripts/setup_and_build.sh", run_content);

    std::cout << "\nProject structure creation finished!\n";
    std::cout << "You can now cd " << root << " && mkdir build && cd build && cmake .. && make\n";
 
    return 0;
}