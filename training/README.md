# Training notes

`yolo_person_detection.ipynb` is the upstream training and conversion workflow. It uses Darknet to train a 96x96 single-class person detector, evaluates the model, converts it to quantized TFLite, and then generates a C array for embedded deployment.

The committed `model/yolo_person_detect.tflite` and `.cc` array are the deployment artifacts used by the Efinix example. Training datasets, Darknet binaries, checkpoints, and generated images are intentionally excluded. If you retrain, replace both deployment artifacts together and re-check input/output quantization, anchors, and accuracy.
