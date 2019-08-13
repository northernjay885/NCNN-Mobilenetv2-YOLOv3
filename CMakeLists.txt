cmake_minimum_required(VERSION 3.15)
project(MobileYOLOv3)
find_package(OpenCV REQUIRED)
# find_package(ncnn REQUIRED)

# set link library and head files of ncnn
set(NCNN_LIBS "/home/northernjay/ncnn/build/install/lib/libncnn.a")
set(NCNN_INCLUDE_DIRS "/home/northernjay/ncnn/build/install/include/ncnn")

# configure OpenMP
find_package(OpenMP REQUIRED)
if(OPENMP_FOUND)
    message("OPENMP FOUND")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()


include_directories(${NCNN_INCLUDE_DIRS})
#include_directories(${OpenCV_INCLUDE_DIR})
#include_directories(${NCNN_INCLUDE_DIRSS})

get_property(dirs DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES)
foreach(dir ${dirs})
  message(STATUS "dir='${dir}'")
endforeach()

add_executable(MobileYOLOv3 mobilenet.cpp)
target_link_libraries(MobileYOLOv3 ${OpenCV_LIBS} ${NCNN_LIBS})
