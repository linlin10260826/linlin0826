#include "yolo_postprocess.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>

namespace {
constexpr float kEps = 1.0e-6f;
inline float sigmoid(float x) { return x >= 0 ? 1.0f/(1.0f+std::exp(-x)) : std::exp(x)/(1.0f+std::exp(x)); }
inline float overlap(float a0,float a1,float b0,float b1) { return std::max(0.0f,std::min(a1,b1)-std::max(a0,b0)); }
inline float iou(const box& a,const box& b) {
    const float inter=overlap(a.x_min,a.x_max,b.x_min,b.x_max)*overlap(a.y_min,a.y_max,b.y_min,b.y_max);
    const float aa=std::max(0.0f,a.x_max-a.x_min)*std::max(0.0f,a.y_max-a.y_min);
    const float ab=std::max(0.0f,b.x_max-b.x_min)*std::max(0.0f,b.y_max-b.y_min);
    return inter/(aa+ab-inter+kEps);
}
inline float score(const box& b) { return b.objectness*(b.class_probabilities ? b.class_probabilities[0] : 1.0f); }
}

extern "C" box* perform_inference(layer* layers,int total,int* total_boxes,int net_height,int net_width,float objectness_threshold,float iou_threshold) {
    if (!layers || total<=0 || !total_boxes || net_height<=0 || net_width<=0) { if(total_boxes)*total_boxes=0; return nullptr; }
    int capacity=0;
    for(int l=0;l<total;++l) capacity += std::max(0,layers[l].height)*std::max(0,layers[l].width)*std::max(0,layers[l].boxes_per_scale);
    box* out=static_cast<box*>(std::calloc(static_cast<size_t>(capacity),sizeof(box)));
    if(!out){*total_boxes=0;return nullptr;}
    int count=0;
    for(int l=0;l<total;++l){
        const layer& in=layers[l];
        const int values=5+in.classes;
        if(!in.outputs||!in.anchors||in.height<=0||in.width<=0||in.boxes_per_scale<=0||in.classes<=0) continue;
        for(int y=0;y<in.height;++y) for(int x=0;x<in.width;++x) for(int a=0;a<in.boxes_per_scale;++a){
            const int base=(y*in.width+x)*in.boxes_per_scale*values+a*values;
            const float obj=sigmoid(in.outputs[base+4]);
            if(obj<objectness_threshold) continue;
            box& b=out[count++];
            const float cx=(x+sigmoid(in.outputs[base]))/static_cast<float>(in.width);
            const float cy=(y+sigmoid(in.outputs[base+1]))/static_cast<float>(in.height);
            const float bw=std::exp(in.outputs[base+2])*in.anchors[a*2]/static_cast<float>(net_width);
            const float bh=std::exp(in.outputs[base+3])*in.anchors[a*2+1]/static_cast<float>(net_height);
            b.x_min=std::max(0.0f,cx-bw*.5f); b.y_min=std::max(0.0f,cy-bh*.5f);
            b.x_max=std::min(1.0f,cx+bw*.5f); b.y_max=std::min(1.0f,cy+bh*.5f); b.objectness=obj;
            b.class_probabilities=static_cast<float*>(std::calloc(static_cast<size_t>(in.classes),sizeof(float)));
            if(!b.class_probabilities){b.objectness=-1.0f;continue;}
            for(int c=0;c<in.classes;++c) b.class_probabilities[c]=sigmoid(in.outputs[base+5+c]);
            b.sort_class_index=0;
        }
    }
    std::sort(out,out+count,[](const box&a,const box&b){return score(a)>score(b);});
    for(int i=0;i<count;++i){
        if(out[i].objectness<0) continue;
        for(int j=i+1;j<count;++j) if(out[j].objectness>=0 && iou(out[i],out[j])>iou_threshold) out[j].objectness=-1.0f;
    }
    int kept=0;
    for(int i=0;i<count;++i){
        if(out[i].objectness>=0){if(kept!=i)out[kept]=out[i];++kept;}
        else std::free(out[i].class_probabilities);
    }
    *total_boxes=kept; return out;
}

extern "C" box* yolo_infer(layer* layers,int total,int* total_boxes,int net_height,int net_width,float objectness_threshold,float iou_threshold){
    return perform_inference(layers,total,total_boxes,net_height,net_width,objectness_threshold,iou_threshold);
}
extern "C" void yolo_free_boxes(box* boxes,int count){if(!boxes)return;for(int i=0;i<count;++i)std::free(boxes[i].class_probabilities);std::free(boxes);}
