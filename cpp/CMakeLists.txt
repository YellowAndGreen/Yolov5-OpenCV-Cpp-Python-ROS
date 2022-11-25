cmake_minimum_required(VERSION 3.22)
project(yolo)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_COMPILER "/usr/local/cuda-11.3/bin/nvcc")
set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda")
set(OpenCV_DIR /home/xu/下载/opencv-4.5.4/build)
set(cv_bridge_DIR "/usr/local/share/cv_bridge/cmake")
find_package(OpenCV 4.5.4 REQUIRED)

add_executable(yolo ./yolo-tiny.cpp)

target_include_directories(yolo PRIVATE
        ${OpenCV_INCLUDE_DIR}
        )

target_link_libraries(yolo
        ${OpenCV_LIBS}
        )
