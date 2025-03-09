#include "kernel_operator.h"
#include <type_traits>      
#define MAX_NUM_BATCHES 1024                         
#define MAX_NUM_CLASSES 1024                 
#define MAX_SPATIAL_DIMENSION 1024   
#define MAX_NUM_BOXES MAX_SPATIAL_DIMENSION                             
#define MAX_SELECTED_INDICES (MAX_NUM_BATCHES * MAX_NUM_CLASSES * MAX_NUM_BOXES)             
using namespace AscendC;                        
constexpr int32_t BUFFER_NUM = 2;                          
template<typename TYPE_boxes, typename TYPE_scores,  typename TYPE_max_output_boxes_per_class, typename TYPE_iou_threshold, typename TYPE_score_threshold, 
        typename TYPE_selected_indices> class KernelNonMaxSuppression_Fast_Fast {                 
    using T = TYPE_boxes;                  
public:                                                
    __aicore__ inline KernelNonMaxSuppression_Fast_Fast() {}                   
    __aicore__ inline void Init(GM_ADDR boxes, GM_ADDR scores, GM_ADDR max_output_boxes_per_class, GM_ADDR iou_threshold, GM_ADDR score_threshold,
                                GM_ADDR selected_indices, GM_ADDR workspace,
                                uint32_t ALIGN_NUM, uint32_t core_size, uint32_t core_remain) {     
        this->blockLength = core_size + core_remain;  
        this->blockLength = this->blockLength + (this->blockLength % ALIGN_NUM ? ALIGN_NUM - this->blockLength % ALIGN_NUM : 0); 
        Gm_boxes.SetGlobalBuffer((__gm__ TYPE_boxes*)boxes,    1*this->blockLength);        
        Gm_scores.SetGlobalBuffer((__gm__ TYPE_scores*)scores, 1*this->blockLength);    
        Gm_max_output_boxes_per_class.SetGlobalBuffer((__gm__ TYPE_max_output_boxes_per_class*)max_output_boxes_per_class, 1); 
        Gm_iou_threshold.SetGlobalBuffer((__gm__ TYPE_iou_threshold*)iou_threshold,       1);  
        Gm_score_threshold.SetGlobalBuffer((__gm__ TYPE_score_threshold*)score_threshold, 1);         
        Gm_selected_indices.SetGlobalBuffer((__gm__ TYPE_selected_indices*)selected_indices, 1*this->blockLength);               
        this->max_output_boxes_per_class = Gm_max_output_boxes_per_class(0);             
        this->iou_threshold = Gm_iou_threshold(0);                
        this->score_threshold = Gm_score_threshold(0);           
        Gm_box_indices.SetGlobalBuffer((__gm__ int32_t*)workspace,  MAX_NUM_BOXES);    
        Gm_box_scores.SetGlobalBuffer((__gm__ float*)workspace,     MAX_NUM_BOXES);    
        Gm_box_coords.SetGlobalBuffer((__gm__ float*)workspace,     MAX_NUM_BOXES*4);                     
        Gm_is_suppressed.SetGlobalBuffer((__gm__ int32_t*)workspace,MAX_NUM_BOXES);          
        pipe.InitBuffer(Q_x, BUFFER_NUM, MAX_NUM_BOXES * 4);         
        pipe.InitBuffer(tmpYBuffer,  MAX_NUM_BOXES * 4);   
        pipe.InitBuffer(tmpDYBuffer, 8*MAX_NUM_BOXES * 4);                            
    }   
    __aicore__ inline void Process(uint32_t shapeInf[2*4], int center_point_box, uint32_t diffNum) {        
        int32_t num_selected_indices = 0;          
        int32_t num_batches = shapeInf[0*4+1];       
        int32_t num_classes = shapeInf[1*4+2];     
        int32_t spatial_dimension= shapeInf[0*4+2]; 
        for (int32_t batch_index = 0; batch_index < num_batches; ++batch_index) {    
            for (int32_t class_index = 0; class_index < num_classes; ++class_index) {    
                int32_t num_boxes = 0;
                for (int32_t box_index = 0; box_index < spatial_dimension; ++box_index) {    
                    float score = Gm_scores(batch_index * num_classes * spatial_dimension + class_index * spatial_dimension + box_index);
                    if (score >= this->score_threshold) {              
                        float y1 = 0.0f, x1 = 0.0f, y2 = 0.0f, x2 = 0.0f;    
                        int32_t index_boxes = batch_index * spatial_dimension * 4 + box_index * 4;
                        y1 = Gm_boxes(index_boxes + 0);   
                        x1 = Gm_boxes(index_boxes + 1);               
                        y2 = Gm_boxes(index_boxes + 2);          
                        x2 = Gm_boxes(index_boxes + 3);       
                        if (y1 > y2) { 
                            float temp = y1;         
                            y1 = y2; 
                            y2 = temp;  
                        }      
                        if (x1 > x2) {
                            float temp = x1;
                            x1 = x2;
                            x2 = temp;
                        }
                        Gm_box_indices(0*MAX_NUM_BOXES+num_boxes) = box_index; 
                        Gm_box_scores(1*MAX_NUM_BOXES+num_boxes) = score; 
                        Gm_box_coords((0+2)*MAX_NUM_BOXES+num_boxes) = y1;     
                        Gm_box_coords((1+2)*MAX_NUM_BOXES+num_boxes) = x1;     
                        Gm_box_coords((2+2)*MAX_NUM_BOXES+num_boxes) = y2;  
                        Gm_box_coords((3+2)*MAX_NUM_BOXES+num_boxes) = x2; 
                        num_boxes++;
                        if (num_boxes >= MAX_NUM_BOXES) break;   
                    }
                }   
                if (num_boxes == 0) continue;                       
                for (int32_t i = 0; i < num_boxes - 1; ++i) {               
                    int32_t max_idx = i;                    
                    for (int32_t j = i + 1; j < num_boxes; ++j) {   
                        if (Gm_box_scores(1*MAX_NUM_BOXES+j) > Gm_box_scores(1*MAX_NUM_BOXES+max_idx)) max_idx = j;   
                    }        
                    float temp_score = Gm_box_scores(1*MAX_NUM_BOXES+i);  
                    Gm_box_scores(1*MAX_NUM_BOXES+i) = Gm_box_scores(1*MAX_NUM_BOXES+max_idx);
                    Gm_box_scores(1*MAX_NUM_BOXES+max_idx) = temp_score;  
                    int32_t temp_index = Gm_box_indices(0*MAX_NUM_BOXES+i);      
                    Gm_box_indices(0*MAX_NUM_BOXES+i) = Gm_box_indices(0*MAX_NUM_BOXES+max_idx);
                    Gm_box_indices(0*MAX_NUM_BOXES+max_idx) = temp_index;  
                    for (int32_t k = 0; k < 4; ++k) {   
                        float temp_coord = Gm_box_coords((k+2)*MAX_NUM_BOXES+i);
                        Gm_box_coords((k+2)*MAX_NUM_BOXES+i) = Gm_box_coords((k+2)*MAX_NUM_BOXES+max_idx);
                        Gm_box_coords((k+2)*MAX_NUM_BOXES+max_idx) = temp_coord;
                    } 
                    Gm_is_suppressed(6*MAX_NUM_BOXES+i) = 0;    
                }       
                Gm_is_suppressed(6*MAX_NUM_BOXES+ num_boxes-1) = 0;           
                int32_t selected_boxes_count = 0;
                for (int32_t i = 0; i < num_boxes; ++i){                    
                    if (Gm_is_suppressed(6*MAX_NUM_BOXES+i)) continue;    
                    if (selected_boxes_count >= this->max_output_boxes_per_class) break;     
                    int32_t selected_index = Gm_box_indices(0*MAX_NUM_BOXES+i);          
                    Gm_selected_indices(num_selected_indices * 3 + 0) = batch_index;
                    Gm_selected_indices(num_selected_indices * 3 + 1) = class_index; 
                    Gm_selected_indices(num_selected_indices * 3 + 2) = selected_index;
                    num_selected_indices++; 
                    selected_boxes_count++;      
                    float y1 = Gm_box_coords((0+2)*MAX_NUM_BOXES+i);
                    float x1 = Gm_box_coords((1+2)*MAX_NUM_BOXES+i); 
                    float y2 = Gm_box_coords((2+2)*MAX_NUM_BOXES+i); 
                    float x2 = Gm_box_coords((3+2)*MAX_NUM_BOXES+i);
                    for (int32_t j = i + 1; j < num_boxes; ++j) {
                        if (Gm_is_suppressed(6*MAX_NUM_BOXES+j)) continue; 
                        float y1_other = Gm_box_coords((0+2)*MAX_NUM_BOXES+j);
                        float x1_other = Gm_box_coords((1+2)*MAX_NUM_BOXES+j);
                        float y2_other = Gm_box_coords((2+2)*MAX_NUM_BOXES+j);
                        float x2_other = Gm_box_coords((3+2)*MAX_NUM_BOXES+j);  
                        float inter_y1 = (y1 > y1_other) ? y1 : y1_other;
                        float inter_x1 = (x1 > x1_other) ? x1 : x1_other;
                        float inter_y2 = (y2 < y2_other) ? y2 : y2_other; 
                        float inter_x2 = (x2 < x2_other) ? x2 : x2_other;
                        float inter_height = inter_y2 - inter_y1; 
                        float inter_width  = inter_x2 - inter_x1;  
                        float inter_area = (inter_height > 0.0f && inter_width > 0.0f) ? inter_height * inter_width : 0.0f; 
                        if (inter_area <= 0.0f) continue;
                        float area1 = ( (y2 - y1) > 0.0f && (x2 - x1) > 0.0f ) ? (y2 - y1) * (x2 - x1) : 0.0f;
                        float area2 = ( (y2_other - y1_other) > 0.0f && (x2_other - x1_other) > 0.0f ) ? (y2_other - y1_other) * (x2_other - x1_other) : 0.0f;
                        float EPSILON = 1e-6f;      
                        if (area1 <= EPSILON || area2 <= EPSILON) continue;
                        float union_area = area1 + area2 - inter_area;
                        float iou = (union_area > EPSILON) ? (inter_area / union_area) : 0.0f;
                        if (iou > this->iou_threshold+ EPSILON)Gm_is_suppressed(6*MAX_NUM_BOXES+j) = 1;   
                    }      
                }     
            }   
        }
    }   
private:    
    TPipe pipe;      
    TQue<QuePosition::VECIN, BUFFER_NUM> Q_x; 
    TBuf<QuePosition::VECCALC> tmpYBuffer, tmpDYBuffer;       
    LocalTensor<float> x;  
    LocalTensor<float> tmpy, tmpdy;     
    GlobalTensor<TYPE_boxes> Gm_boxes;        
    GlobalTensor<TYPE_scores> Gm_scores;   
    GlobalTensor<TYPE_max_output_boxes_per_class> Gm_max_output_boxes_per_class;         
    GlobalTensor<TYPE_iou_threshold> Gm_iou_threshold;   
    GlobalTensor<TYPE_score_threshold> Gm_score_threshold;       
    GlobalTensor<int32_t> Gm_box_indices;          
    GlobalTensor<float> Gm_box_scores;               
    GlobalTensor<float> Gm_box_coords;             
    GlobalTensor<int32_t> Gm_is_suppressed;     
    GlobalTensor<TYPE_selected_indices> Gm_selected_indices;    
    uint32_t blockLength;    
    float iou_threshold;
    float score_threshold; 
    int32_t max_output_boxes_per_class;    
};   

extern "C" __global__ __aicore__ void non_max_suppression(GM_ADDR boxes, GM_ADDR scores, GM_ADDR max_output_boxes_per_class, GM_ADDR iou_threshold, GM_ADDR score_threshold,
                                            GM_ADDR selected_indices,   
                                            GM_ADDR workspace, GM_ADDR tiling) { 
    GET_TILING_DATA(tiling_data, tiling);              
    KernelNonMaxSuppression_Fast_Fast<DTYPE_BOXES, DTYPE_SCORES, DTYPE_MAX_OUTPUT_BOXES_PER_CLASS, DTYPE_IOU_THRESHOLD, DTYPE_SCORE_THRESHOLD, DTYPE_SELECTED_INDICES> op;      
    op.Init(boxes, scores, max_output_boxes_per_class, iou_threshold,score_threshold,selected_indices, workspace,
            tiling_data.ALIGN_NUM, tiling_data.core_size, tiling_data.core_remain);         
    op.Process(tiling_data.shapeInf,tiling_data.center_point_box,
               tiling_data.diffNum);         
    
}   

