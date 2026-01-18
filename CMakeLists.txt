
cmake_minimum_required(VERSION 3.16)
project(cat_feeder VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

add_executable(cat_feeder
  src/main.cpp
  src/camera.cpp
  src/detector.cpp
  src/perf.cpp
  src/postprocessor.cpp
  src/preprocessor.cpp
)

target_link_libraries(cat_feeder PRIVATE ${OpenCV_LIBS})
target_include_directories(cat_feeder PRIVATE ${OpenCV_INCLUDE_DIRS})
