- # LapSRN2TensorRT

> 本项目是将基于Pytorch的LapSRN图像超分模型部署到TensorRT，技术路线为“pytorch model-->onnx file-->TensorRT engine”。
>
> 说明：当前只针对ONNX和TensorRT支持OP可进行转换。

## 软件环境：

```
Python3.8.5
TensorRT7.2.2.1
Pytorch1.8
PIL8.1.0
numpy1.19.5
pycuda2020.1
Linux_x86_64
CUDA11.1
CUDNN8.0.0
```

## 当前支持：

-  TensorRT FP32
-  TensorRT FP16
-  TensorRT INT8

## 使用方法：

1. 从Pytorch模型到ONNX：修改并使用pytorch2onnx.py脚本转ONNX。
2. 根据上一步转换好的ONNX文件，进行TensorRT转换，使用命令 trtexec --verbose --onnx=results/lapsrn.onnx --saveEngine=./results/lapsrn_fp32.trt，trtexec --verbose --onnx=lapsrn.onnx --saveEngine=./results/lapsrn_fp16.trt --fp16 进行fp32和fp16转换。
3. INT8量化可参考int8量化说明。
4. 使用`do_inference.py`进行推理验证。可使用原始训练模型、fp32、fp16、int8进行单张图片推理测试，详细结果可见test_results。
5. 速度对比：运行 engine_speed_compare.py 脚本进行engine文件与原始模型的速度对比。

## int8量化说明：

Pytorch模型转ONNX：

- 参考脚本pytorch2onnx.py，需按照自己的需要定义模型与输入样例，然后转换。

将ONNX转换为INT8的TensorRT引擎，需要:

1. 图像超分重建校准集calibrator_data，大小是240*240。用于在转换过程中寻找使得转换后的激活值分布与原来的FP32类型的激活值分布差异最小的阈值;

2. 一个校准器类，该类继承trt.IInt8EntropyCalibrator2父类，具体内容参考脚本`myCalibrator.py`.

3. 使用时，需额外指定cache_file，这里指定为lapsrn.cache , 该参数是校准集cache文件的路径，会在校准过程中生成，方便下一次校准时快速提取。

   

   ## 参考：

   https://github.com/NVIDIA/trt-samples-for-hackathon-cn

   https://github.com/qq995431104/Pytorch2TensorRT#pytorch2tensorrt

   

   

   