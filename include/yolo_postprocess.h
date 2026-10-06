#pragma once
#ifdef __cplusplus
extern "C" {
#endif
typedef struct yolo_layer { int channels,height,width,classes,boxes_per_scale,total_anchors; float scale; const float* anchors; float* outputs; } yolo_layer;
typedef struct yolo_box { float x_min,y_min,x_max,y_max,objectness; float* class_probabilities; int sort_class_index; } yolo_box;
typedef yolo_layer layer;
typedef yolo_box box;
box* perform_inference(layer* layers,int total,int* total_boxes,int net_height,int net_width,float objectness_threshold,float iou_threshold);
box* yolo_infer(layer* layers,int total,int* total_boxes,int net_height,int net_width,float objectness_threshold,float iou_threshold);
void yolo_free_boxes(box* boxes,int count);
#ifdef __cplusplus
}
#endif
