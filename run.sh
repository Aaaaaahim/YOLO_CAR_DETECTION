#!/bin/bash
# YOLOv8n 推理运行脚本
# 自动编译并运行 OpenCV DNN 版本

# 切换到脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# 设置颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 自动检测 cmake 路径
if command -v cmake &> /dev/null; then
    CMAKE_CMD="cmake"
elif [ -f "/home/him/miniconda3/envs/rknn2/bin/cmake" ]; then
    CMAKE_CMD="/home/him/miniconda3/envs/rknn2/bin/cmake"
else
    echo -e "${RED}错误: 未找到 cmake，请安装 CMake${NC}"
    exit 1
fi

# 检查是否需要重新编译
NEED_BUILD=0
if [ ! -d "build" ] || [ ! -f "build/yolov8n_infer" ]; then
    NEED_BUILD=1
fi

# 如果传入 --rebuild 参数，强制重新编译
if [ "$1" == "--rebuild" ]; then
    NEED_BUILD=1
    shift  # 移除 --rebuild 参数
fi

# 编译项目
if [ $NEED_BUILD -eq 1 ]; then
    echo -e "${YELLOW}正在编译项目...${NC}"
    
    # 创建 build 目录
    mkdir -p build
    cd build
    
    # 运行 CMake 配置
    $CMAKE_CMD .. > /dev/null 2>&1
    if [ $? -ne 0 ]; then
        echo -e "${RED}CMake 配置失败！${NC}"
        exit 1
    fi
    
    # 编译
    make -j$(nproc) > /dev/null 2>&1
    if [ $? -ne 0 ]; then
        echo -e "${RED}编译失败！请检查代码错误。${NC}"
        exit 1
    fi
    
    cd ..
    echo -e "${GREEN}编译成功！${NC}"
else
    echo -e "${GREEN}使用已编译的程序（使用 --rebuild 强制重新编译）${NC}"
fi

# 设置库路径（使用项目内部的第三方库）
export LD_LIBRARY_PATH="$SCRIPT_DIR/3rdparty/opencv/lib:$SCRIPT_DIR/3rdparty/protobuf/lib:$SCRIPT_DIR/3rdparty/abseil/lib:$SCRIPT_DIR/3rdparty/jpeg/lib:$SCRIPT_DIR/lib:$LD_LIBRARY_PATH"

# 检查必要的文件
if [ ! -f "yolov8n_sim.onnx" ]; then
    echo -e "${RED}错误: 找不到模型文件 yolov8n_sim.onnx${NC}"
    echo "请确保模型文件在项目根目录下"
    exit 1
fi

# 运行程序
echo -e "${GREEN}正在运行推理...${NC}"

# 默认参数
INPUT_IMG="test.jpeg"
OUTPUT_IMG="result.jpg"

# 解析参数
if [ $# -ge 1 ]; then
    INPUT_IMG="$1"
fi
if [ $# -ge 2 ]; then
    OUTPUT_IMG="$2"
fi

# 显示运行参数
echo "  模型文件: yolov8n_sim.onnx"
echo "  输入图片: $INPUT_IMG"
echo "  输出图片: $OUTPUT_IMG"

# 检查输入文件是否存在
if [ ! -f "$INPUT_IMG" ]; then
    echo -e "${RED}错误: 找不到输入图片 [$INPUT_IMG]${NC}"
    echo "用法: $0 [输入图片] [输出图片]"
    exit 1
fi

# 运行程序
./build/yolov8n_infer yolov8n_sim.onnx "$INPUT_IMG" "$OUTPUT_IMG" 80

# 检查运行结果
if [ $? -eq 0 ]; then
    echo -e "${GREEN}推理完成！${NC}"
else
    echo -e "${RED}推理失败！${NC}"
    exit 1
fi
