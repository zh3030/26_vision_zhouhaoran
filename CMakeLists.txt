cmake_minimum_required(VERSION 3.10)
project(BuffEnergyDetector VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui videoio)

# ---------- 功能开关 ----------
option(ENABLE_TEST        "Enable debug visualization"         ON)
option(ENABLE_BUFF_TIME   "Enable step timing output"          ON)
option(ENABLE_ADD_POINTS  "Enable sector center points output" ON)
option(ENABLE_WRITE       "Enable video writing output"        OFF)

# ---------- 可执行文件 ----------
add_executable(buff_detector
    main.cpp
    src/buff_detector.cpp
)

# ---------- 包含路径 ----------
target_include_directories(buff_detector PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
)

# ---------- 链接库 ----------
target_link_libraries(buff_detector PRIVATE ${OpenCV_LIBS})

# ---------- 传递宏定义 ----------
if(ENABLE_TEST)
    target_compile_definitions(buff_detector PRIVATE BUFF_TEST)
endif()
if(ENABLE_BUFF_TIME)
    target_compile_definitions(buff_detector PRIVATE BUFF_TIME)
endif()
if(ENABLE_ADD_POINTS)
    target_compile_definitions(buff_detector PRIVATE ADD_POINTS)
endif()
if(ENABLE_WRITE)
    target_compile_definitions(buff_detector PRIVATE BUFF_WRITE)
endif()