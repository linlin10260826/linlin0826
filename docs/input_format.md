# 输入约定

模型输入是 NHWC 排列的 96*96*3 个 int8 值。官方静态样例已经完成量化，数组长度为 27648。

实时输入适配时，先把摄像头帧缩放或中心裁剪到 96x96，再按模型的 input quantization 参数计算：

`q = round(real_value / scale) + zero_point`

并将 q 限制在 [-128,127]。`scale` 和 `zero_point` 必须从当前 `.tflite` 模型读取，不能凭经验固定。

输出是两个 int8 YOLO 层。先按各输出 tensor 自己的 scale/zero_point 反量化：

`real_value = (q - zero_point) * scale`

再将每层组织成 `yolo_layer`，调用 `perform_inference`。输出框坐标在 0 到 1 的相对坐标范围内，类别 0 是 person。
