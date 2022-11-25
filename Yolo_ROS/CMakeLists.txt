cmake_minimum_required(VERSION 3.10)
project(yolo_ros)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CUDA_COMPILER "/usr/local/cuda-11.3/bin/nvcc")
set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda")
set(OpenCV_DIR /home/xu/下载/opencv-4.5.4/build)
set(cv_bridge_DIR /usr/local/share/cv_bridge/cmake)

find_package(OpenCV 4.5.4 REQUIRED)

find_package(catkin REQUIRED COMPONENTS
        image_transport
        roscpp
        rospy
        cv_bridge
        sensor_msgs
        std_msgs
        )

include_directories(include ${catkin_INCLUDE_DIRS} )

add_executable(yolo_ros ./ros.cpp)

target_include_directories(yolo_ros PRIVATE
        ${OpenCV_INCLUDE_DIR}
        )

target_link_libraries(yolo_ros
        ${OpenCV_LIBS}
        )

target_link_libraries(yolo_ros
        ${catkin_LIBRARIES}
        )

target_link_libraries(yolo_ros /opt/ros/melodic/lib/libmessage_filters.so)
