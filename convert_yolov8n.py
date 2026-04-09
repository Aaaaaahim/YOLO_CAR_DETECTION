from rknn.api import RKNN

ONNX_MODEL = 'yolov8n_sim.onnx'
DATASET_TXT = 'dataset.txt'
RKNN_MODEL = 'yolov8n.rknn'

def main():
    # 如果不设置verbose_file就将日志消息打印到终端,否则设置打印到文件内
    # rknn = RKNN(verbose=True,verbose_file='log.txt')
    rknn = RKNN(verbose=True)

    print('1) Config RKNN')
    ret = rknn.config(
        target_platform='RV1106'  # 指定目标平台
    )
    # 判断是否设置模型的预处理参数成功
    if ret != 0:
        print('config failed')
        return
    
    print('2) Load ONNX')
    ret = rknn.load_onnx(model=ONNX_MODEL)
    if ret != 0:
        print('load onnx failed')
        return
    
    print('3) Build RKNN')
    # 核心步骤，用于将加载的深度学习模型（如ONNX、TensorFlow、PyTorch等转换
    # 为RKNN格式，并进行针对Rockchip NPU的优化和量化处理。
    ret = rknn.build(
        # 进行量化（True 表示量化为 int8，False 表示保持浮点模型）
        do_quantization=True,  
        dataset=DATASET_TXT
    )
    if ret != 0:
        print('build failed')
        return

    print('4) Export RKNN')
    # RKNN-Toolkit2 提供的 API，用于将已经通过 build() 转换好的模型
    # 导出为 RKNN 格式文件（.rknn），以便在瑞芯微 NPU 设备上部署运行
    ret = rknn.export_rknn(RKNN_MODEL)
    if ret != 0:
        print('export failed')
        return

    rknn.release()
    print('done')

if __name__ == '__main__':
    main()