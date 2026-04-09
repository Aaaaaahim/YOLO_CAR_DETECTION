cmake_minimum_required(VERSION 3.10)
project(yolov8n_rknn)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 获取脚本所在目录（项目根目录）
get_filename_component(PROJECT_ROOT "${CMAKE_SOURCE_DIR}" ABSOLUTE)

include_directories(${PROJECT_ROOT}/include)
include_directories(${PROJECT_ROOT}/3rdparty/opencv/include)

# 使用相对路径的第三方库
set(OPENCV_LIBS_DIR ${PROJECT_ROOT}/3rdparty/opencv/lib)
set(JPEG_LIB_DIR ${PROJECT_ROOT}/3rdparty/jpeg/lib)
set(PROTOBUF_LIB_DIR ${PROJECT_ROOT}/3rdparty/protobuf/lib)
set(ABSL_LIB_DIR ${PROJECT_ROOT}/3rdparty/abseil/lib)
set(RKNN_LIB_DIR ${PROJECT_ROOT}/lib)

# ONNX 推理版本 (使用 OpenCV DNN)
add_executable(yolov8n_infer src/main.cpp)

target_link_libraries(yolov8n_infer 
    ${OPENCV_LIBS_DIR}/libopencv_core.so
    ${OPENCV_LIBS_DIR}/libopencv_imgproc.so
    ${OPENCV_LIBS_DIR}/libopencv_imgcodecs.so
    ${OPENCV_LIBS_DIR}/libopencv_dnn.so
    ${PROTOBUF_LIB_DIR}/libprotobuf.so
    ${ABSL_LIB_DIR}/libabsl_log_internal_check_op.so.2401.0.0
    ${ABSL_LIB_DIR}/libabsl_log_internal_message.so.2401.0.0
    ${ABSL_LIB_DIR}/libabsl_log_internal_nullguard.so.2401.0.0
    ${ABSL_LIB_DIR}/libabsl_hash.so.2401.0.0
    ${ABSL_LIB_DIR}/libabsl_spinlock_wait.so.2401.0.0
    ${JPEG_LIB_DIR}/libjpeg.so
)

set_target_properties(yolov8n_infer PROPERTIES 
    BUILD_RPATH "$ORIGIN/../3rdparty/opencv/lib:$ORIGIN/../3rdparty/protobuf/lib:$ORIGIN/../3rdparty/abseil/lib:$ORIGIN/../3rdparty/jpeg/lib:$ORIGIN/../lib"
)

# RKNN 版本 (使用 RKNN API)
add_executable(yolov8n_rknn src/main_rknn.cpp)

target_link_libraries(yolov8n_rknn 
    ${OPENCV_LIBS_DIR}/libopencv_core.so
    ${OPENCV_LIBS_DIR}/libopencv_imgproc.so
    ${OPENCV_LIBS_DIR}/libopencv_imgcodecs.so
    ${JPEG_LIB_DIR}/libjpeg.so
    ${RKNN_LIB_DIR}/librknn_api.so
)

set_target_properties(yolov8n_rknn PROPERTIES 
    BUILD_RPATH "$ORIGIN/../3rdparty/opencv/lib:$ORIGIN/../3rdparty/jpeg/lib:$ORIGIN/../lib"
)
