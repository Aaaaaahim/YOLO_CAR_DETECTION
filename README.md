# YOLOv8n 汽车检测系统

基于 YOLOv8n 的实时汽车检测项目，支持 ONNX 和 RKNN 两种推理方式，可在 PC 端和瑞芯微 NPU 设备上运行。

## 📋 项目简介

本项目实现了基于 YOLOv8n 模型的目标检测系统，专注于汽车检测场景。提供了完整的模型转换、推理和可视化流程，支持：

- **PC 端推理**：使用 OpenCV DNN 模块加载 ONNX 模型
- **NPU 加速推理**：使用 RKNN-Toolkit2 将模型转换为 RKNN 格式，在瑞芯微 NPU 上运行
- **多语言实现**：提供 C++ 实现方式

## ✨ 主要特性

- ✅ 支持 YOLOv8n 模型的 ONNX 到 RKNN 格式转换
- ✅ 完整的图像预处理流程（Letterbox、归一化）
- ✅ 高效的后处理算法（NMS、置信度过滤）
- ✅ 支持 COCO 80 类目标检测
- ✅ 可视化检测结果（边界框、类别标签、置信度）
- ✅ 针对瑞芯微 RV1106/RK3588 平台优化

## 🛠️ 技术栈

- **深度学习框架**：YOLOv8n
- **模型格式**：ONNX、RKNN
- **推理引擎**：OpenCV DNN、RKNN API
- **编程语言**：Python 3.8+、C++11
- **构建工具**：CMake 3.10+
- **图像处理**：OpenCV 4.10.0

## 📁 项目结构

```
rknn_data/
├── src/                          # 源代码目录
│   ├── main.cpp                  # OpenCV DNN 推理实现 (C++)
│   └── main_rknn.cpp             # RKNN API 推理实现 (C++)
├── include/                      # RKNN API 头文件目录
│   ├── rknn_api.h
│   ├── rknn_custom_op.h
│   └── rknn_matmul_api.h
├── lib/                          # RKNN 库文件目录
│   └── librknn_api.so            # RKNN API 动态库
├── 3rdparty/                     # 第三方依赖库（项目自包含）
│   ├── opencv/                   # OpenCV 库
│   │   ├── include/              # OpenCV 头文件
│   │   └── lib/                  # OpenCV 动态库
│   ├── protobuf/                 # Protobuf 库
│   │   └── lib/
│   ├── abseil/                   # Abseil 库
│   │   └── lib/
│   └── jpeg/                     # JPEG 库
│       └── lib/
├── datasets/                     # 量化数据集
│   ├── 1.jpeg
│   ├── 2.jpeg
│   └── ...                       # 11 张校准图片
├── build/                        # CMake 构建目录
├── convert_yolov8n.py            # ONNX 转 RKNN 模型脚本
├── dataset.txt                   # 量化数据集路径列表
├── CMakeLists.txt                # CMake 配置文件
├── run.sh                        # OpenCV DNN 版本运行脚本
├── run_rknn.sh                   # RKNN 版本运行脚本
├── yolov8n_sim.onnx              # 简化后的 ONNX 模型
├── yolov8n.rknn                  # 转换后的 RKNN 模型
└── test.jpeg                     # 测试图片
```

## 🚀 快速开始

### 环境要求

#### PC 端开发环境
- Ubuntu 18.04+ / Linux
- Python 3.8+
- CMake 3.10+
- GCC/G++ 7.5+

**注意**：本项目已包含所有必要的 C++ 依赖库（OpenCV、Protobuf、Abseil、JPEG），无需额外安装系统依赖。

#### NPU 设备环境
- 瑞芯微 RV1106 / RK3588 开发板
- RKNN-Toolkit2 1.5.0+
- RKNN API 库（已包含在项目中）

### 安装依赖

#### 1. 安装 Python 依赖

```bash
# 安装 RKNN-Toolkit2
pip install rknn-toolkit2

# 安装 OpenCV
pip install opencv-python numpy
```

#### 2. C++ 依赖（已包含）

本项目已自带所有 C++ 依赖库，无需额外安装。
如需重新编译，只需确保已安装 CMake 和 GCC：
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install build-essential cmake
```

### 模型准备

#### 1. 获取 YOLOv8n ONNX 模型（已获取）

```bash
# 使用 Ultralytics 导出 ONNX 模型
pip install ultralytics
yolo export model=yolov8n.pt format=onnx simplify=True imgsz=640
```

将导出的 `yolov8n.onnx` 重命名为 `yolov8n_sim.onnx` 并放置在项目根目录。

#### 2. 转换为 RKNN 模型（可选/已获取）

如果需要在瑞芯微 NPU 上运行，执行模型转换：

```bash
python convert_yolov8n.py
```

转换过程包括：
1. 加载 ONNX 模型
2. 配置目标平台（RV1106）
3. INT8 量化（使用 datasets/ 中的图片）
4. 导出 RKNN 模型

**注意**：量化过程需要提供代表性数据集，项目已包含 11 张校准图片。

## 💻 使用方法

### C++ 推理

#### 1. 编译项目（自动）

项目使用自包含的第三方库，直接编译即可：

```bash
# OpenCV DNN 版本
./run.sh --rebuild

# RKNN API 版本
./run_rknn.sh --rebuild
```

手动编译：
```bash
mkdir -p build
cd build
cmake ..
make
```

编译会生成两个可执行文件：
- `yolov8n_infer`：OpenCV DNN 版本
- `yolov8n_rknn`：RKNN API 版本

#### 2. 运行推理

##### 参数说明

- `输入图片`：待检测的图片路径
- `输出图片`：保存检测结果的路径

**OpenCV DNN 版本**：

```bash
# 使用快速运行脚本
./run.sh [输入图片] [输出图片]

# 或手动运行
./build/yolov8n_infer yolov8n_sim.onnx test.jpeg result.jpg 80
```

**RKNN 版本**：

```bash
./run_rknn.sh [输入图片] [输出图片]
# 或手动运行
./build/yolov8n_rknn yolov8n.rknn test.jpeg result.jpg
```

## 🔧 配置说明

### 检测参数调整

在源代码中可以调整以下参数：

#### C++ 版本 (`src/main.cpp`)

```cpp
#define INPUT_SIZE      640     // 输入尺寸
#define CONF_THRESHOLD  0.6f    // 置信度阈值
#define IOU_THRESHOLD   0.45f   // NMS IoU 阈值
#define TARGET_CLASS    2       // 目标类别 (2=car)
```

## 📊 性能指标

### 模型信息

| 模型 | 输入尺寸 | 参数量 | ONNX 大小 | RKNN 大小 |
|------|---------|--------|-----------|-----------|
| YOLOv8n | 640×640 | 3.2M | 12.2 MB | 12.4 MB |

### 推理性能（参考）

| 平台 | 推理方式 | 推理时间 | FPS |
|------|---------|---------|-----|
| PC (CPU) | OpenCV DNN | ~50ms | ~20 |
| RV1106 (NPU) | RKNN API | ~15ms | ~66 |
| RK3588 (NPU) | RKNN API | ~8ms | ~125 |

*注：实际性能取决于具体硬件配置和图像分辨率*

## 🎯 检测类别

项目支持 COCO 数据集的 80 个类别，常见类别包括：

| ID | 类别 | ID | 类别 | ID | 类别 |
|----|------|----|----- |----|------|
| 0 | person | 1 | bicycle | 2 | car |
| 3 | motorcycle | 5 | bus | 7 | truck |

完整类别列表请参考 `COCO_NAMES` 数组。

## 🔍 算法流程

### 1. 图像预处理

```
原始图像 → Letterbox 填充 → 归一化 → 模型输入
```

- **Letterbox**：保持宽高比缩放到 640×640，空白区域填充灰色 (114, 114, 114)
- **归一化**：像素值除以 255，转换为 [0, 1] 范围

### 2. 模型推理

```
输入 [1, 3, 640, 640] → YOLOv8n → 输出 [1, 84, 8400]
```

- 输出格式：`[batch, 4+80, anchors]`
  - 前 4 个通道：边界框坐标 (x, y, w, h)
  - 后 80 个通道：类别置信度

### 3. 后处理

```
模型输出 → Sigmoid 激活 → 置信度过滤 → 坐标转换 → NMS → 最终结果
```

- **置信度过滤**：保留 score > CONF_THRESHOLD 的检测
- **坐标转换**：xywh → xyxy，还原 Letterbox 变换
- **NMS**：去除重叠的检测框（IoU > IOU_THRESHOLD）

## 🐛 常见问题

### 1. 编译错误：找不到 OpenCV

**解决方案**：
```bash
# 使用 --rebuild 参数重新编译
./run.sh --rebuild
```

### 2. 运行时错误：找不到共享库

**解决方案**：
run.sh 脚本已自动设置库路径。如需手动设置：
```bash
export LD_LIBRARY_PATH=./3rdparty/opencv/lib:./3rdparty/protobuf/lib:./3rdparty/abseil/lib:./3rdparty/jpeg/lib:$LD_LIBRARY_PATH
```

### 3. RKNN 转换失败

**可能原因**：
- ONNX 模型未简化
- 量化数据集不足或质量差
- 目标平台配置错误

**解决方案**：
```bash
# 确保 ONNX 模型已简化
pip install onnx-simplifier
python -m onnxsim yolov8n.onnx yolov8n_sim.onnx

# 增加量化数据集图片（建议 100-200 张）
# 确保图片内容与实际应用场景相似
```

### 4. 检测结果不准确

**调整建议**：
- 降低 `CONF_THRESHOLD` 增加召回率
- 调整 `IOU_THRESHOLD` 控制重叠框
- 使用更多样化的量化数据集
- 考虑使用更大的模型（YOLOv8s/m）

## 📝 开发说明

### 添加自定义类别

1. 修改 `COCO_NAMES` 数组
2. 调整 `TARGET_CLASS` 为目标类别 ID
3. 重新训练模型（如需要）

### 优化推理速度

1. **模型量化**：使用 INT8 量化减少计算量
2. **输入尺寸**：降低输入分辨率（如 320×320）
3. **后处理优化**：减少 TOPK 数量，提高 CONF_THRESHOLD
4. **多线程**：使用 OpenMP 并行化后处理

### 移植到其他平台

1. 修改 `CMakeLists.txt` 中的库路径
2. 调整 `convert_yolov8n.py` 中的 `target_platform`
3. 根据平台特性优化预处理流程

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

## 🙏 致谢

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) - YOLOv8 模型
- [RKNN-Toolkit2](https://github.com/rockchip-linux/rknn-toolkit2) - 瑞芯微 NPU 工具链
- [OpenCV](https://opencv.org/) - 计算机视觉库

## 📧 联系方式

如有问题或建议，请通过以下方式联系：

- 提交 [Issue]
- 发送邮件至：[him676739@gmail.com]

---

⭐ 如果这个项目对你有帮助，请给个 Star！
