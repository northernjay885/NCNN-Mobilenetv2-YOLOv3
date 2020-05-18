# NCNN-Mobilenetv2-YOLOv3

Build a YOLOv3 Convolutional Neural Network for object detection using NCNN framework.

## Getting Started

```
git clone https://github.com/northernjay885/NCNN-Mobilenetv2-YOLOv3.git
```

### Prerequisites


Install [CMake](https://cmake.org/download/).

### Installing

Build OpenCV using CMake
```
cmake -D CMAKE_BUILD_TYPE=Release -D CMAKE_INSTALL_PREFIX=/usr/local ..
```
```
make -j8 # runs 8 jobs in parallel, depend on your CPU threads
```
Build NCNN framework following the [guide](https://github.com/Tencent/ncnn).

## Build objects

in /build directory
```
cmake -D CMAKE_BUILD_TYPE=Release ..
```

## Built With

* [Cmake](https://www.djangoproject.com/)
* [OpenCV](https://opencv.org/)
* [NCNN](https://github.com/Tencent/ncnn)
