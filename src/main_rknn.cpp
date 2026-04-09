#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <algorithm>
#include <cmath>
#include <opencv2/opencv.hpp>

#include "rknn_api.h"

#define INPUT_SIZE 640
#define CONF_THRESHOLD 0.65f
#define IOU_THRESHOLD 0.4f
#define TOPK 100

// COCO 类别名称
const char* coco_names[] = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "dondonut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush"
};

struct Detection {
    float x1, y1, x2, y2;
    float score;
    int class_id;
};

// Letterbox 预处理
void letterbox(cv::Mat& img, cv::Mat& out, float& scale, int& pad_w, int& pad_h) {
    int h = img.rows;
    int w = img.cols;
    
    scale = std::min((float)INPUT_SIZE / h, (float)INPUT_SIZE / w);
    int new_w = (int)(w * scale);
    int new_h = (int)(h * scale);
    
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);
    
    pad_w = (INPUT_SIZE - new_w) / 2;
    pad_h = (INPUT_SIZE - new_h) / 2;
    
    out = cv::Mat(INPUT_SIZE, INPUT_SIZE, CV_8UC3, cv::Scalar(114, 114, 114));
    resized.copyTo(out(cv::Rect(pad_w, pad_h, new_w, new_h)));
}

// Sigmoid 函数
float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// 计算 IoU
float compute_iou(Detection& a, Detection& b) {
    float x1 = std::max(a.x1, b.x1);
    float y1 = std::max(a.y1, b.y1);
    float x2 = std::min(a.x2, b.x2);
    float y2 = std::min(a.y2, b.y2);
    
    float inter_area = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
    float area_a = (a.x2 - a.x1) * (a.y2 - a.y1);
    float area_b = (b.x2 - b.x1) * (b.y2 - b.y1);
    float union_area = area_a + area_b - inter_area;
    
    return (union_area > 0) ? (inter_area / union_area) : 0.0f;
}

// NMS 非极大值抑制
std::vector<Detection> nms(std::vector<Detection>& boxes, float iou_threshold) {
    std::sort(boxes.begin(), boxes.end(), [](const Detection& a, const Detection& b) {
        return a.score > b.score;
    });
    
    std::vector<bool> suppressed(boxes.size(), false);
    std::vector<Detection> result;
    
    for (size_t i = 0; i < boxes.size(); i++) {
        if (suppressed[i]) continue;
        
        result.push_back(boxes[i]);
        
        for (size_t j = i + 1; j < boxes.size(); j++) {
            if (suppressed[j]) continue;
            if (boxes[j].class_id != boxes[i].class_id) continue;
            
            if (compute_iou(boxes[i], boxes[j]) > iou_threshold) {
                suppressed[j] = true;
            }
        }
    }
    
    return result;
}

// 后处理
std::vector<Detection> postprocess(float* output, int output_size, 
                                   int orig_w, int orig_h,
                                   float scale, int pad_w, int pad_h) {
    // YOLOv8 输出: (1, 84, 8400) -> 8400 个预测框，每个 84 维 (4 bbox + 80 classes)
    const int num_proposals = 8400;
    const int num_classes = 80;
    
    std::vector<Detection> detections;
    
    for (int i = 0; i < num_proposals; i++) {
        float* ptr = output + i * (4 + num_classes);
        
        float cx = ptr[0];
        float cy = ptr[1];
        float w = ptr[2];
        float h = ptr[3];
        
        // 找最大置信度的类别
        float max_score = 0;
        int class_id = 0;
        for (int c = 0; c < num_classes; c++) {
            float score = sigmoid(ptr[4 + c]);
            if (score > max_score) {
                max_score = score;
                class_id = c;
            }
        }
        
        // 置信度过滤
        if (max_score < CONF_THRESHOLD) continue;
        
        // xywh -> xyxy
        Detection det;
        det.x1 = cx - w * 0.5f;
        det.y1 = cy - h * 0.5f;
        det.x2 = cx + w * 0.5f;
        det.y2 = cy + h * 0.5f;
        det.score = max_score;
        det.class_id = class_id;
        
        detections.push_back(det);
    }
    
    // TopK 过滤
    if ((int)detections.size() > TOPK) {
        std::partial_sort(detections.begin(), detections.begin() + TOPK, detections.end(),
            [](const Detection& a, const Detection& b) { return a.score > b.score; });
        detections.resize(TOPK);
    }
    
    // 还原 letterbox padding
    for (auto& det : detections) {
        det.x1 -= pad_w;
        det.y1 -= pad_h;
        det.x2 -= pad_w;
        det.y2 -= pad_h;
        
        // 还原到原图尺寸
        det.x1 /= scale;
        det.y1 /= scale;
        det.x2 /= scale;
        det.y2 /= scale;
        
        // 裁剪到原图范围
        det.x1 = std::max(0.0f, std::min((float)orig_w - 1, det.x1));
        det.y1 = std::max(0.0f, std::min((float)orig_h - 1, det.y1));
        det.x2 = std::max(0.0f, std::min((float)orig_w - 1, det.x2));
        det.y2 = std::max(0.0f, std::min((float)orig_h - 1, det.y2));
    }
    
    // NMS
    return nms(detections, IOU_THRESHOLD);
}

// 绘制结果
void draw_detections(cv::Mat& img, std::vector<Detection>& detections) {
    for (const auto& det : detections) {
        cv::rectangle(img, 
            cv::Point(det.x1, det.y1), 
            cv::Point(det.x2, det.y2), 
            cv::Scalar(0, 255, 0), 2);
        
        std::string label = coco_names[det.class_id];
        label += " " + std::to_string(det.score).substr(0, 4);
        
        int baseline;
        cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.6, 2, &baseline);
        
        cv::rectangle(img,
            cv::Point(det.x1, det.y1 - text_size.height - 5),
            cv::Point(det.x1 + text_size.width, det.y1),
            cv::Scalar(0, 255, 0), -1);
        
        cv::putText(img, label, cv::Point(det.x1, det.y1 - 5),
            cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 2);
    }
}

int main(int argc, char** argv) {
    const char* model_path = "yolov8n.rknn";
    const char* img_path = "test.jpg";
    const char* output_path = "result.jpg";
    
    if (argc > 1) model_path = argv[1];
    if (argc > 2) img_path = argv[2];
    if (argc > 3) output_path = argv[3];
    
    printf("=== YOLOv8n RKNN C++ Inference ===\n");
    printf("Model: %s\n", model_path);
    printf("Image: %s\n", img_path);
    
    // 1. 读取图像
    cv::Mat img = cv::imread(img_path);
    if (img.empty()) {
        printf("Error: Cannot load image %s\n", img_path);
        return -1;
    }
    printf("Image shape: %dx%d\n", img.cols, img.rows);
    
    int orig_w = img.cols;
    int orig_h = img.rows;
    
    // 2. Letterbox 预处理
    cv::Mat input_img;
    float scale;
    int pad_w, pad_h;
    letterbox(img, input_img, scale, pad_w, pad_h);
    printf("Preprocessed to: %dx%d, scale=%.3f, pad=(%d,%d)\n", 
           INPUT_SIZE, INPUT_SIZE, scale, pad_w, pad_h);
    
    // 3. 初始化 RKNN
    rknn_context ctx;
    
    // 读取模型文件到内存
    FILE* fp = fopen(model_path, "rb");
    if (!fp) {
        printf("Error: Cannot open model file: %s\n", model_path);
        return -1;
    }
    fseek(fp, 0, SEEK_END);
    long model_size = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    
    char* model_data = (char*)malloc(model_size);
    if (!model_data) {
        printf("Error: Cannot allocate memory for model\n");
        fclose(fp);
        return -1;
    }
    
    if (fread(model_data, 1, model_size, fp) != (size_t)model_size) {
        printf("Error: Cannot read model file\n");
        fclose(fp);
        free(model_data);
        return -1;
    }
    fclose(fp);
    
    printf("Loading model from memory (%ld bytes)...\n", model_size);
    int ret = rknn_init(&ctx, model_data, model_size, RKNN_FLAG_PRIOR_HIGH, NULL);
    free(model_data);
    
    if (ret < 0) {
        printf("Error: rknn_init failed! ret=%d\n", ret);
        return -1;
    }
    printf("RKNN initialized successfully\n");
    
    // 4. 查询模型信息
    rknn_input_output_num io_num;
    ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret < 0) {
        printf("Error: rknn_query RKNN_QUERY_IN_OUT_NUM failed!\n");
        rknn_destroy(ctx);
        return -1;
    }
    printf("Model inputs: %d, outputs: %d\n", io_num.n_input, io_num.n_output);
    
    // 5. 设置输入
    rknn_input inputs[1];
    memset(inputs, 0, sizeof(inputs));
    inputs[0].index = 0;
    inputs[0].buf = input_img.data;
    inputs[0].size = INPUT_SIZE * INPUT_SIZE * 3;
    inputs[0].pass_through = 0;
    inputs[0].type = RKNN_TENSOR_UINT8;
    inputs[0].fmt = RKNN_TENSOR_NHWC;
    
    ret = rknn_inputs_set(ctx, 1, inputs);
    if (ret < 0) {
        printf("Error: rknn_inputs_set failed! ret=%d\n", ret);
        rknn_destroy(ctx);
        return -1;
    }
    
    // 6. 推理
    ret = rknn_run(ctx, NULL);
    if (ret < 0) {
        printf("Error: rknn_run failed! ret=%d\n", ret);
        rknn_destroy(ctx);
        return -1;
    }
    printf("Inference completed\n");
    
    // 7. 获取输出
    rknn_output outputs[1];
    memset(outputs, 0, sizeof(outputs));
    outputs[0].want_float = 1;
    outputs[0].is_prealloc = 0;
    
    ret = rknn_outputs_get(ctx, 1, outputs, NULL);
    if (ret < 0) {
        printf("Error: rknn_outputs_get failed! ret=%d\n", ret);
        rknn_destroy(ctx);
        return -1;
    }
    
    // 8. 后处理
    float* output_ptr = (float*)outputs[0].buf;
    int output_size = outputs[0].size / sizeof(float);
    printf("Output size: %d floats\n", output_size);
    
    std::vector<Detection> detections = postprocess(output_ptr, output_size, 
                                                    orig_w, orig_h, scale, pad_w, pad_h);
    printf("Detections: %zu\n", detections.size());
    
    // 9. 绘制结果
    draw_detections(img, detections);
    cv::imwrite(output_path, img);
    printf("Result saved to: %s\n", output_path);
    
    // 打印检测结果
    for (const auto& det : detections) {
        printf("  [%s] score=%.2f, box=(%.1f, %.1f, %.1f, %.1f)\n",
               coco_names[det.class_id], det.score, 
               det.x1, det.y1, det.x2, det.y2);
    }
    
    // 10. 释放资源
    rknn_outputs_release(ctx, 1, outputs);
    rknn_destroy(ctx);
    
    printf("Done!\n");
    return 0;
}
