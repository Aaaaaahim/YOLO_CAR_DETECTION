/**
 * YOLOv8n 汽车检测程序 (OpenCV DNN 版本)
 * 
 * 功能: 使用 OpenCV DNN 模块加载 ONNX 模型进行目标检测
 * 适用: Ubuntu PC 环境下测试推理流程
 * 
 * 使用方法:
 *   ./yolov8n_infer [模型路径] [图片路径] [输出路径] [类别数]
 *   例如: ./yolov8n_infer yolov8n_sim.onnx test.jpg result.jpg 80
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <algorithm>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <iomanip>

//==============================================================================
// 配置文件 - 根据实际需求修改这些参数
//==============================================================================

#define INPUT_SIZE      640     // 模型输入尺寸 (YOLOv8n 默认 640x640)
#define CONF_THRESHOLD  0.6f    // 置信度阈值,高于此值的检测结果才会保留
#define IOU_THRESHOLD   0.45f   // NMS IoU 阈值,用于去除重叠的检测框
#define TARGET_CLASS    2       // 目标类别 ID (COCO 数据集中 car=2)

// COCO 类别名称对照表
const char* CLASS_NAMES[] = 
{
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck"
};

//==============================================================================
// 数据结构
//==============================================================================

/**
 * 检测结果结构体
 * 存储单个目标检测的边界框和置信度
 */
class Detection 
{
public:
    float x1, y1;    // 左上角坐标
    float x2, y2;    // 右下角坐标
    float score;      // 置信度分数
};

//==============================================================================
// 工具函数
//==============================================================================

/**
 * Sigmoid 激活函数
 * 将数值转换为 0-1 之间的概率值
 */
inline float sigmoid(float x) 
{
    return 1.0f / (1.0f + expf(-x));
}

/**
 * 计算两个边界框的 IoU (Intersection over Union)
 * 用于 NMS 算法中判断框的重叠程度
 * 
 * @param a 第一个检测框
 * @param b 第二个检测框
 * @return IoU 值,范围 [0, 1]
 */
float compute_iou(const Detection& a, const Detection& b) 
{
    /*
    y轴向下递增
    y=0  ─────────────────→ x
    │    
    │    框A: (100, 50) ────┐
    │         │              │
    │         │    框B: (150, 80) ────┐
    │         │         │              │
    │         └─────────┼──── (200, 150)
    │                   │              │
    │                   └────────────── (250, 180)
    ↓
    y
    */
    // 计算交集区域
    float x1 = std::max(a.x1, b.x1);
    float y1 = std::max(a.y1, b.y1);
    float x2 = std::min(a.x2, b.x2);
    float y2 = std::min(a.y2, b.y2);
    
    float inter_area = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
    
    // 计算各自面积
    float area_a = (a.x2 - a.x1) * (a.y2 - a.y1);
    float area_b = (b.x2 - b.x1) * (b.y2 - b.y1);
    
    // 计算并集面积
    float union_area = area_a + area_b - inter_area;
    
    // 返回交集对总面积的所占比例
    return (union_area > 0) ? (inter_area / union_area) : 0.0f;
}

//==============================================================================
// 图像预处理
//==============================================================================

/**
 * Letterbox 图像缩放
 * 保持原图宽高比,将图像缩放到目标尺寸,不足部分用灰色填充
 * 
 * @param img    输入图像
 * @param out    输出图像 (640x640)
 * @param scale  缩放比例 (输出参数)
 * @param pad_w  水平填充像素数 (输出参数)
 * @param pad_h  垂直填充像素数 (输出参数)
 */
void letterbox(const cv::Mat& img, cv::Mat& out, float& scale, int& pad_w, int& pad_h) 
{
    int h = img.rows;
    int w = img.cols;
    
    // 计算缩放比例,保持宽高比
    scale = std::min((float)INPUT_SIZE / h, (float)INPUT_SIZE / w);
    
    // 计算缩放后的尺寸
    int new_w = (int)(w * scale);
    int new_h = (int)(h * scale);
    
    // 缩放图像
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);
    
    // 计算填充量,使图像居中 (左上角的坐标) 
    pad_w = (INPUT_SIZE - new_w) / 2;
    pad_h = (INPUT_SIZE - new_h) / 2;
    
    // 创建 640x640 的灰色画布,将缩放后的图像放入中央
    out = cv::Mat(INPUT_SIZE, INPUT_SIZE, CV_8UC3, cv::Scalar(114, 114, 114));
    resized.copyTo(out(cv::Rect(pad_w, pad_h, new_w, new_h)));  // Rect：左上角坐标+宽度与高度   Mat(...) -> 截取矩形区域
}

//==============================================================================
// NMS (非极大值抑制)
//==============================================================================

/**
 * 非极大值抑制 (NMS)
 * 去除重叠的检测框,保留置信度最高的框
 * 
 * @param boxes        检测框列表
 * @param iou_threshold IoU 阈值,超过此值的框会被抑制
 * @return 保留的检测框列表
 */
std::vector<Detection> nms(std::vector<Detection>& boxes, float iou_threshold) {
    if (boxes.empty()) return {};
    
    // 按置信度从高到低排序
    std::sort(boxes.begin(), boxes.end(), [](const Detection& a, const Detection& b) {
        return a.score > b.score;
    });
    
    std::vector<bool> suppressed(boxes.size(), false);  // 标记数组，true 表示该框已被抑制（删除）
    std::vector<Detection> result;
    
    // 遍历所有框
    for (size_t i = 0; i < boxes.size(); i++) 
    {
        if (suppressed[i]) continue;
        
        // 保留当前框
        result.push_back(boxes[i]);
        
        // 抑制与当前框 IoU 超过阈值的框
        for (size_t j = i + 1; j < boxes.size(); j++) 
        {
            if (suppressed[j]) continue;
            if (compute_iou(boxes[i], boxes[j]) > iou_threshold) 
            {
                // 如果后续存在重叠程度满足时，就把j号的设为true（删除），保留置信度更高的i号框
                suppressed[j] = true;
            }
        }
    }
    
    return result;
}

//==============================================================================
// 模型输出后处理
//==============================================================================

/**
 * YOLOv8n 输出后处理
 * 将模型原始输出转换为检测结果
 * 
 * 
 * YOLOv8 输出格式: (1, 84, 8400)
 * - 1: batch size
 * - 84: 4(边界框) + 80(类别数)
 * - 8400: 预测框数量
 * 
 * 需要经过 transpose 转换为: (8400, 84),每行是一个预测
 * 
 * @param output          模型输出,形状 (1, 84, 8400)
 * @param orig_w          原图宽度
 * @param orig_h          原图高度
 * @param scale           缩放比例
 * @param pad_w           水平填充
 * @param pad_h           垂直填充
 * @param num_classes     类别数
 * 
 * @return  处理后的检测框
 */
std::vector<Detection> postprocess(
    const cv::Mat& output,     // 模型输出,形状 (1, 84, 8400)
    int orig_w,                // 原图宽度
    int orig_h,                // 原图高度
    float scale,               // 缩放比例
    int pad_w,                 // 水平填充
    int pad_h,                 // 垂直填充
    int num_classes            // 类别数
) {
    std::vector<Detection> detections;
    
    // 1. 转换输出格式: (1, 84, 8400) -> (84, 8400) -> (8400, 84)
    
    // 步骤 1.1: reshape - 去除 batch 维度，将三维张量变为二维矩阵
    // - 参数1: channels=1，强制视为单通道数据
    // - 参数2: {84, 8400}，新的形状为 84 行 × 8400 列
    // - 结果: 每一列代表一个预测框的 84 个特征值
    // - 注意: reshape 不会复制数据，只是改变数据的"视图"，效率很高
    cv::Mat reshaped = output.reshape(1, {output.size[1], output.size[2]});  // (84, 8400)
    
    // 步骤 1.2: transpose - 转置矩阵，行列互换
    // - 将 (84, 8400) 转置为 (8400, 84)
    // - 结果: 每一行代表一个预测框，包含 84 个特征值 [cx, cy, w, h, score1, ..., score80]
    // - 目的: 方便后续按"框"为单位遍历处理（每次循环处理一个完整的框）
    cv::Mat transposed;
    cv::transpose(reshaped, transposed);  // (8400, 84)
    
    // 2. 遍历每个预测框
    for (int i = 0; i < transposed.rows; i++) 
    {
        float* ptr = transposed.ptr<float>(i);
        
        // 解析边界框坐标 (YOLOv8 格式: cx, cy, w, h)
        float cx = ptr[0];
        float cy = ptr[1];
        float w  = ptr[2];
        float h  = ptr[3];
        
        // 3. 找到所有类别中置信度最高的
        // [cx, cy, w, h, score_0, score_1, score_2, ..., score_79]
        /*
        后 80 个（索引 4-83）：80 个 COCO 类别的原始得分 
            ptr[4]：person（人）的得分 
            ptr[5]：bicycle（自行车）的得分 
            ptr[6]：car（汽车）的得分  
            ptr[83]：第 80 个类别的得分
        */
        float max_score = 0;
        int class_id = 0;
        for (int c = 0; c < num_classes; c++) 
        {
            float score = sigmoid(ptr[4 + c]);
            if (score > max_score) 
            {
                max_score = score;
                class_id = c;
            }
        }
        
        // 4. 只保留目标类别且置信度超过阈值的检测
        if (class_id != TARGET_CLASS) continue;
        if (max_score < CONF_THRESHOLD) continue;
        
        // 5. 过滤无效框 (宽高过小或坐标异常)
        if (w <= 1 || h <= 1) continue;
        if (cx < 0 || cy < 0 || cx > INPUT_SIZE * 2 || cy > INPUT_SIZE * 2) continue;
        
        // 6. 坐标转换: (cx, cy, w, h) -> (x1, y1, x2, y2)
        Detection det;
        det.x1 = cx - w * 0.5f;
        det.y1 = cy - h * 0.5f;
        det.x2 = cx + w * 0.5f;
        det.y2 = cy + h * 0.5f;
        det.score = max_score;
        
        detections.push_back(det);
    }
    
    std::cout<< "[后处理] NMS 前检测数:"<<detections.size()<<std::endl;
    
    // 7. 还原坐标到原图尺寸
    for (auto& det : detections) 
    {
        // 去除 padding
        det.x1 -= pad_w;
        det.y1 -= pad_h;
        det.x2 -= pad_w;
        det.y2 -= pad_h;
        
        // 还原到原图尺寸
        det.x1 /= scale;
        det.y1 /= scale;
        det.x2 /= scale;
        det.y2 /= scale;
        
        // 裁剪到原图范围内
        det.x1 = std::max(0.0f, std::min((float)orig_w - 1, det.x1));
        det.y1 = std::max(0.0f, std::min((float)orig_h - 1, det.y1));
        det.x2 = std::max(0.0f, std::min((float)orig_w - 1, det.x2));
        det.y2 = std::max(0.0f, std::min((float)orig_h - 1, det.y2));
    }
    
    // 8. 执行 NMS
    return nms(detections, IOU_THRESHOLD);
}

//==============================================================================
// 结果绘制
//==============================================================================

/**
 * 在图像上绘制检测结果
 * 
 * @param img        原始图像 (会直接修改)
 * @param detections 检测结果列表
 */
void draw_detections(cv::Mat& img, const std::vector<Detection>& detections) 
{
    // 遍历所有检测结果
    for (const auto& det : detections) 
    {
        // 跳过无效框
        if (det.x2 <= det.x1 || det.y2 <= det.y1) continue;
        
        // 绘制边界框 (绿色,线宽2)
        cv::rectangle(img, 
            cv::Point(det.x1, det.y1), 
            cv::Point(det.x2, det.y2), 
            cv::Scalar(0, 255, 0), 2);
        
        // 绘制标签背景
        char label[64];
        snprintf(label, sizeof(label), "%s %.2f", CLASS_NAMES[TARGET_CLASS], det.score);
        
        cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.6, 2, 0);
        
        cv::rectangle(img,
            cv::Point(det.x1, det.y1 - text_size.height - 5),
            cv::Point(det.x1 + text_size.width, det.y1),
            cv::Scalar(0, 255, 0), -1);  // -1 表示填充
        
        // 绘制标签文字 (白色)
        cv::putText(img, label, cv::Point(det.x1, det.y1 - 5),
            cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);
    }
}

//==============================================================================
// 主函数
//==============================================================================

int main(int argc, char** argv) 
{
    //-------------------------------------------------------------------------
    // 命令行参数解析
    //-------------------------------------------------------------------------
    const char* onnx_path    = "yolov8n_sim.onnx";  // 模型文件路径
    const char* img_path     = "test.jpg";           // 输入图片路径
    const char* output_path  = "result.jpg";         // 输出图片路径
    int num_classes          = 80;                    // 类别数 (COCO=80)
    
    // 支持命令行参数覆盖
    if (argc > 1) onnx_path = argv[1];
    if (argc > 2) img_path = argv[2];
    if (argc > 3) output_path = argv[3];
    if (argc > 4) num_classes = atoi(argv[4]);
    
    std::cout<<"========================================"<<std::endl;
    std::cout<<"   YOLOv8n 汽车检测 (OpenCV DNN 版本)"<<std::endl;
    std::cout<<"========================================"<<std::endl;
    std::cout<<"模型文件:"<<onnx_path<<std::endl;
    std::cout<<"输入图片:"<<img_path<<std::endl;
    std::cout<<"目标类别:"<<CLASS_NAMES[TARGET_CLASS]<<"(ID="<<TARGET_CLASS<<")"<<std::endl;
    std::cout<<"----------------------------------------"<<std::endl;
    
    //-------------------------------------------------------------------------
    // 1. 加载图像
    //-------------------------------------------------------------------------
    cv::Mat img = cv::imread(img_path);
    if (img.empty()) 
    {
        std::cout<<"错误: 无法加载图片 ["<<img_path<<"]"<<std::endl;
        return -1;
    }
    
    int orig_w = img.cols;
    int orig_h = img.rows;
    std::cout<<"[步骤1] 加载图片: "<<orig_w<<"X"<<orig_h<<std::endl;
    
    //-------------------------------------------------------------------------
    // 2. 图像预处理 (Letterbox)
    //-------------------------------------------------------------------------
    cv::Mat input_img;
    float scale;
    int pad_w, pad_h;

    letterbox(img, input_img, scale, pad_w, pad_h);
    // printf("[步骤2] 预处理: 缩放=%.3f, 填充=(%d,%d)\n", scale, pad_w, pad_h);
    std::cout<<"[步骤2] 预处理：缩放="<<std::setprecision(3)<<scale<<"，"<<"填充=("<<pad_w<<","<<pad_h<<")"<<std::endl;
    
    //-------------------------------------------------------------------------
    // 3. 加载 ONNX 模型
    //-------------------------------------------------------------------------
    /*
     *加载模型：支持 .pb（TensorFlow）、.onnx（ONNX）、.caffemodel（Caffe）等格式
     *预处理：cv::dnn::blobFromImage() 将图像转换为网络输入张量（blob）
     *推理：net.forward() 获取模型输出
     *硬件加速：支持 CPU、OpenCL、CUDA（需编译支持）
    */
    cv::dnn::Net net = cv::dnn::readNetFromONNX(onnx_path);
    if (net.empty()) 
    {
        std::cout<<"错误：无法加载模型 ["<<onnx_path<<"]"<<std::endl;
        return -1;
    }
    std::cout<<"[步骤3] 模型加载成功"<<std::endl;
    
    //-------------------------------------------------------------------------
    // 4. 准备模型输入
    //-------------------------------------------------------------------------
    // 归一化 + 尺寸调整 + 通道转换 (HWC -> NCHW)
    cv::Mat blob = cv::dnn::blobFromImage(
        input_img, 
        1.0 / 255.0,              // 归一化到 [0, 1]
        cv::Size(INPUT_SIZE, INPUT_SIZE),  // 目标尺寸
        cv::Scalar(0, 0, 0),      // 不做均值减法
        true,                      // RGB -> BGR 转换
        false                      // 不裁剪
    );
    net.setInput(blob);
    std::cout<<"[步骤4] 输入准备完成，blob shape: "<<cv::format("[%dx%d]", blob.size[2], blob.size[3]).c_str()<<std::endl;
    
    //-------------------------------------------------------------------------
    // 5. 执行推理
    //-------------------------------------------------------------------------
    std::cout<<"[步骤5] 正在推理..."<<std::endl;
    std::vector<cv::Mat> outputs;
    /*
    forward: DNN模块中，是用于通过加载神经网络进行前向传递并获得一个或多个层输出的方法
    getUnconnectedOutLayersNames: 获取当前网络中未连接的输出层的名称列表
    */
    net.forward(outputs, net.getUnconnectedOutLayersNames());
    
    if (outputs.empty()) 
    {
        printf("错误: 模型无输出\n");
        return -1;
    }
    
    cv::Mat output = outputs[0];
    std::cout<<"       输出形状: ["<<output.size[0]<<"X"<<output.size[1]<<"X"<<output.size[2]<<"]"<<std::endl;
    
    //-------------------------------------------------------------------------
    // 6. 后处理
    //-------------------------------------------------------------------------
    std::vector<Detection> detections = postprocess(
        output, orig_w, orig_h, scale, pad_w, pad_h, num_classes
    );
    std::cout<<"[步骤6] 检测完成：共 "<<detections.size()<<" 个目标"<<std::endl;
    
    //-------------------------------------------------------------------------
    // 7. 绘制并保存结果
    //-------------------------------------------------------------------------
    draw_detections(img, detections);
    cv::imwrite(output_path, img);
    std::cout<<"[步骤7] 结果已保存："<<output_path<<std::endl;
    
    // 打印检测结果详情
    std::cout<<"----------------------------------------"<<std::endl;
    std::cout<<"检测结果："<<std::endl;
    for (size_t i = 0; i < std::min((size_t)10, detections.size()); i++) 
    {
        const auto& d = detections[i];
        printf("  [%zu] %s: %.2f, 位置: (%.0f, %.0f, %.0f, %.0f)\n",
               i + 1, CLASS_NAMES[TARGET_CLASS], d.score, d.x1, d.y1, d.x2, d.y2);
    }

    std::cout<<"========================================"<<std::endl;
    std::cout<<"完成！"<<std::endl;
    
    return 0;
}
