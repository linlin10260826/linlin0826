# YOLO Pico 行人检测算法

这是从 Efinix `tinyml-main` 示例中整理出的行人检测算法分支，面向 Ti60F225 上的 TinyML/TFLite Micro 集成。

## 工程边界

本目录提供模型、输入量化约定、YOLO 输出解码和 NMS 后处理。摄像头采集、DDR3 帧缓存、HDMI 输出、RISC-V 启动文件以及 FPGA bitstream 属于平台集成层，需要在 `platform/efinix_ti60` 中接入。当前目录不等同于完整的摄像头产品程序。

处理链：摄像头 RGB 帧 -> 96x96 -> int8 输入 -> TFLite Micro 推理 -> 两个 YOLO 输出层 -> anchor 解码 -> 阈值 -> NMS -> 行人框。

## 模型和输入

* 模型：`model/yolo_person_detect.tflite`
* 输入：`96 x 96 x 3`、`int8`。官方样例输入已预量化到 `[-128,127]`。
* `model/input_img_000000018380_96x96x3.cc` 是官方静态测试输入。
* `model/yolo_person_detect_model_data.cc` 是无文件系统嵌入式目标使用的模型数组。
* 原始 `.tflite` 模型大小为 231,368 字节。

普通 RGB 像素必须使用模型的 input scale 和 zero point 转换为 int8；不能直接把 `[0,255]` 写入 tensor。

## 后处理参数

```text
classes = 1                 # person
anchors = [115,73], [119,199], [242,238]
          [12,18],  [37,49],  [52,132]
objectness threshold = 0.25
iou threshold = 0.5
```

`src/yolo_postprocess.cc` 保留原 `yolo.h` 的核心数据结构和 `perform_inference` 接口，提供可读的 sigmoid、anchor 解码和 NMS 实现。它使用 TFLite 输出的 NHWC 展开顺序。

## Efinix 集成

`platform/efinix_ti60/reference_main.cc` 说明官方程序如何加载模型、写入 int8 输入、调用 `Invoke()`，再把两个输出层反量化后交给后处理。该文件依赖 Efinix BSP、TinyML accelerator 和 TensorFlow Lite Micro，不能在普通电脑上直接编译。

当前目录验证的是算法结构和依赖边界；摄像头、HDMI、Ti60F225 启动镜像仍需按实际板卡工程适配。

## 许可证和来源

原工程采用 MIT License，见 `NOTICE.md`。模型和数组文件来自 Efinix `github-efx/tinyml-main` 示例；公开发布前请保留来源并核对模型权重的再分发条件。
