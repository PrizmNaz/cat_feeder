
cmake_minimum_required(VERSION 3.16)
project(cat_feeder VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

find_package(OpenCV REQUIRED)
find_package(Threads REQUIRED)

add_executable(cat_feeder
  src/main.cpp
  src/camera.cpp
  src/detector.cpp
  src/perf.cpp
  src/postprocessor.cpp
  src/preprocessor.cpp
  src/web_server.cpp
)

target_link_libraries(cat_feeder PRIVATE ${OpenCV_LIBS} Threads::Threads)
target_include_directories(cat_feeder PRIVATE ${OpenCV_INCLUDE_DIRS})

add_executable(preprocess_recorder
  src/preprocess_recorder.cpp
  src/camera.cpp
  src/preprocessor.cpp
)

target_link_libraries(preprocess_recorder PRIVATE ${OpenCV_LIBS} Threads::Threads)
target_include_directories(preprocess_recorder PRIVATE ${OpenCV_INCLUDE_DIRS})

option(CAT_FEEDER_RPI4_OPT "Enable Raspberry Pi 4 compile flags on ARM64" ON)
if (CAT_FEEDER_RPI4_OPT AND CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|armv8")
  target_compile_options(cat_feeder PRIVATE -mcpu=cortex-a72 -mtune=cortex-a72)
  target_compile_options(preprocess_recorder PRIVATE -mcpu=cortex-a72 -mtune=cortex-a72)
endif()
