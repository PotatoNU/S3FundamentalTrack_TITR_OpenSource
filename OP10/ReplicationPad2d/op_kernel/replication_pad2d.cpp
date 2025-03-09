#include "kernel_operator.h"
#include <type_traits>           

using namespace AscendC;     
constexpr int32_t BUFFER_NUM = 2;     
template<typename T> struct Map {using type = T;};           
template<> struct Map<int8_t> {using type = half;};              
template<typename TYPE_X, typename TYPE_PADDINGS,  typename TYPE_Y> class Kernel_Broadcast {
    using T = TYPE_Y;           
public:     
    __aicore__ inline Kernel_Broadcast() {}     
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR paddings, GM_ADDR y, 
                                uint32_t ALIGN_NUM, uint32_t block_size, uint32_t core_size, uint32_t core_remain) {    
        this->blockLength = core_size + core_remain;  
        this->blockLength = this->blockLength + (this->blockLength % ALIGN_NUM ? ALIGN_NUM - this->blockLength % ALIGN_NUM : 0); 
        Gm_x.SetGlobalBuffer((__gm__ TYPE_X*)x , this->blockLength);
        Gm_paddings.SetGlobalBuffer((__gm__ TYPE_PADDINGS*)paddings , this->blockLength);
        Gm_y.SetGlobalBuffer((__gm__ TYPE_Y*)y , this->blockLength);     
    }    
    __aicore__ inline void Process(uint32_t shapeInf[1*5]) {        
        if (shapeInf[0*5+0] == 3) {  
            int max_index[3] = {};
            for (int j = 1; j <= shapeInf[0]; j++) {
                if (shapeInf[j] > max_index[j - 1]) { 
                    max_index[j - 1] = shapeInf[j];
                }
            }
            int depth = max_index[0];
            int height = max_index[1];  
            int width = max_index[2];
            int padding_left   = Gm_paddings(0);
            int padding_right  = Gm_paddings(1);
            int padding_top    = Gm_paddings(2);
            int padding_bottom = Gm_paddings(3);
            int padded_height = height + padding_top + padding_bottom;
            int padded_width  = width + padding_left + padding_right; 
            for (int d = 0; d < depth; d++) {   
                for (int i = 0; i < height; i++) {
                    for (int j = 0; j < width; j++) {
                        Gm_y((d * padded_height + i + padding_top) * padded_width + (j + padding_left)) = 
                            Gm_x((d * height + i) * width + j);
                    }
                }
                for (int j = 0; j < width; j++) {
                    for (int p = 0; p < padding_top; p++) {
                        Gm_y((d * padded_height + p) * padded_width + (j + padding_left)) = 
                            Gm_x((d * height + 0) * width + j); 
                    }
                    for (int p = 0; p < padding_bottom; p++) {
                        Gm_y((d * padded_height + height + padding_top + p) * padded_width + (j + padding_left)) = 
                            Gm_x((d * height + height - 1) * width + j);
                    }
                }
                for (int i = 0; i < height; i++) {
                    for (int p = 0; p < padding_left; p++) {
                        Gm_y((d * padded_height + i + padding_top) * padded_width + p) = 
                            Gm_x((d * height + i) * width + 0); 
                    }
                    for (int p = 0; p < padding_right; p++) {
                        Gm_y((d * padded_height + i + padding_top) * padded_width + width + padding_left + p) = 
                            Gm_x((d * height + i) * width + width - 1); 
                    }
                }
                for (int p = 0; p < padding_top; p++) {
                    for (int q = 0; q < padding_left; q++) {
                        Gm_y((d * padded_height + p) * padded_width + q) = 
                            Gm_x((d * height + 0) * width + 0); 
                    }
                    for (int q = 0; q < padding_right; q++) {
                        Gm_y((d * padded_height + p) * padded_width + (width + padding_left + q)) = 
                            Gm_x((d * height + 0) * width + width - 1); 
                    }
                }
                for (int p = 0; p < padding_bottom; p++) {
                    for (int q = 0; q < padding_left; q++) {
                        Gm_y((d * padded_height + height + padding_top + p) * padded_width + q) = 
                            Gm_x((d * height + height - 1) * width + 0);
                    }
                    for (int q = 0; q < padding_right; q++) {
                        Gm_y((d * padded_height + height + padding_top + p) * padded_width + (width + padding_left + q)) = 
                            Gm_x((d * height + height - 1) * width + width - 1); 
                    }
                }
            }
        }else if (shapeInf[0*5+0] == 4) {   
            int max_index[4] = {};
            for (int j = 1; j <= shapeInf[0]; j++) {
                if (shapeInf[j] > max_index[j - 1]) {
                    max_index[j - 1] = shapeInf[j];
                }
            } 
            int batch = max_index[0];
            int depth = max_index[1];
            int height = max_index[2];
            int width = max_index[3];
            int padding_left   = Gm_paddings(0);
            int padding_right  = Gm_paddings(1);
            int padding_top    = Gm_paddings(2);
            int padding_bottom = Gm_paddings(3);
            int padded_height = height + padding_top + padding_bottom;
            int padded_width  = width + padding_left + padding_right;
            for (int b = 0; b < batch; b++) {
                for (int d = 0; d < depth; d++) {
                    for (int i = 0; i < height; i++) {
                        for (int j = 0; j < width; j++) {
                            Gm_y((b * depth * padded_height + d * padded_height + i + padding_top) * padded_width + (j + padding_left)) = 
                                Gm_x((b * depth * height + d * height + i) * width + j);
                        }
                    }
                    for (int j = 0; j < width; j++) {
                        for (int p = 0; p < padding_top; p++) {
                            Gm_y((b * depth * padded_height + d * padded_height + p) * padded_width + (j + padding_left)) = 
                                Gm_x((b * depth * height + d * height + 0) * width + j); 
                        }
                        for (int p = 0; p < padding_bottom; p++) {
                            Gm_y((b * depth * padded_height + d * padded_height + height + padding_top + p) * padded_width + (j + padding_left)) = 
                                Gm_x((b * depth * height + d * height + height - 1) * width + j);
                        }
                    }   
                    for (int i = 0; i < height; i++) {
                        for (int p = 0; p < padding_left; p++) {
                            Gm_y((b * depth * padded_height + d * padded_height + i + padding_top) * padded_width + p) = 
                                Gm_x((b * depth * height + d * height + i) * width + 0);
                        }
                        for (int p = 0; p < padding_right; p++) {
                            Gm_y((b * depth * padded_height + d * padded_height + i + padding_top) * padded_width + width + padding_left + p) = 
                                Gm_x((b * depth * height + d * height + i) * width + width - 1); 
                        }
                    }
                    for (int p = 0; p < padding_top; p++) {
                        for (int q = 0; q < padding_left; q++) {
                            Gm_y((b * depth * padded_height + d * padded_height + p) * padded_width + q) = 
                                Gm_x((b * depth * height + d * height + 0) * width + 0);
                        }
                        for (int q = 0; q < padding_right; q++) {
                            Gm_y((b * depth * padded_height + d * padded_height + p) * padded_width + (width + padding_left + q)) = 
                                Gm_x((b * depth * height + d * height + 0) * width + width - 1);
                        }
                    }
                    for (int p = 0; p < padding_bottom; p++) {
                        for (int q = 0; q < padding_left; q++) {
                            Gm_y((b * depth * padded_height + d * padded_height + height + padding_top + p) * padded_width + q) = 
                                Gm_x((b * depth * height + d * height + height - 1) * width + 0); 
                        }
                        for (int q = 0; q < padding_right; q++) {
                            Gm_y((b * depth * padded_height + d * padded_height + height + padding_top + p) * padded_width + (width + padding_left + q)) = 
                                Gm_x((b * depth * height + d * height + height - 1) * width + width - 1); 
                        }
                    }
                }
            }
        }
    }    
private: 
    GlobalTensor<TYPE_X> Gm_x; 
    GlobalTensor<TYPE_PADDINGS> Gm_paddings; 
    GlobalTensor<TYPE_Y> Gm_y;   
    uint32_t blockLength;    
};
 
// 两阶段思路
template<typename TYPE_X, typename TYPE_PADDINGS,  typename TYPE_Y> class Kernel_Broadcast_Fast_V4 { 
    using T = TYPE_Y;                 
public:
    __aicore__ inline Kernel_Broadcast_Fast_V4() {}       
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR paddings, GM_ADDR y,        
                                uint32_t ALIGN_NUM, uint32_t block_size, uint32_t core_size, uint32_t core_remain,       
                                uint32_t shapeInf[1*5]){           
        this->blockLength = core_size + core_remain; 
        this->blockLength = this->blockLength + (this->blockLength % ALIGN_NUM ? ALIGN_NUM - this->blockLength % ALIGN_NUM : 0);    
        Gm_x.SetGlobalBuffer((__gm__ TYPE_X*)x , this->blockLength);     
        Gm_paddings.SetGlobalBuffer((__gm__ TYPE_PADDINGS*)paddings , 4);                 
        this->type_num = 9;    
        batch  = shapeInf[1];        
        depth  = shapeInf[2]; 
        height = shapeInf[3];     
        width  = shapeInf[4];   
        padding_left   = Gm_paddings(0);   
        padding_right  = Gm_paddings(1);     
        padding_top    = Gm_paddings(2);       
        padding_bottom = Gm_paddings(3);     
        padded_height = height + padding_top  + padding_bottom;        
        padded_width  = width  + padding_left + padding_right;     
        this->tileLength = shapeInf[4];     
        this->blockLength_y = batch*depth*padded_height*padded_width;                    
        this->blockLength_y = (this->blockLength_y % (8*ALIGN_NUM)) ? (this->blockLength_y/(8*ALIGN_NUM)*(8*ALIGN_NUM)+(8*ALIGN_NUM)) : (this->blockLength_y+(8*ALIGN_NUM));         
        Gm_y.SetGlobalBuffer((__gm__ TYPE_Y*)y , this->blockLength_y);                                      
    }           
    __aicore__ inline void Process(uint32_t shapeInf[1*5]) {  
        /*
            思路：GM_X->左右边->中心块-> GM_Y
                  GM_Y->上下边
        */  
        pipe.InitBuffer(Q_x,  BUFFER_NUM, height*width * 4);                
        pipe.InitBuffer(Q_y,  BUFFER_NUM, height*width * 4);           
        for (int b = 0; b < batch; b++) {      
            for (int d = 0; d < depth; d++) {       
                index_x = b * depth * height + d * height;  
                index_y = b * depth * padded_height + d * padded_height;     
                {         
                    x = Q_x.AllocTensor<TYPE_X>();             
                    DataCopyExtParams copyParams{1, static_cast<uint32_t>(height*width*4), 0, 0, 0};              
                    DataCopyPadExtParams<TYPE_X> padParams{true, 0, 0, 0};      
                    DataCopyPad(x, Gm_x[(index_x + 0) * width + 0], copyParams, padParams);   
                    Q_x.EnQue(x);     
                }       
                x = Q_x.DeQue<TYPE_X>();       
                y = Q_y.AllocTensor<TYPE_Y>();      
                DataCopyExtParams copyParams{static_cast<uint16_t>(height),    
                                             static_cast<uint32_t>(8*4),      
                                             0,                                    
                                             static_cast<uint32_t>((padding_left + padding_right + 128-8) * 4), 
                                             0};          
                {                                         
                    uint64_t mask[2] = {0x80, 0};                            
                    Muls(y,x[(16-1)*8],static_cast<TYPE_Y>(1), mask, 128, { 0, 0, 1, 2*8 });                                                  
                    Q_y.EnQue<TYPE_Y>(y);   
                    y_tmp = Q_y.DeQue<TYPE_Y>();             
                }                      
                for (int i = padding_left+128; i >=padding_left+128-7; i--){                                    
                    DataCopyPad(Gm_y[(index_y + 0 + padding_top) * padded_width + (i)], y_tmp[0], copyParams);                                                    
                }
                {                
                    Muls(y,x,static_cast<TYPE_Y>(1), 1, 128, { 0, 0, 1, 2*8 });       
                    Q_x.FreeTensor(x);                    
                    Q_y.EnQue<TYPE_Y>(y); 
                    y_tmp = Q_y.DeQue<TYPE_Y>();                  
                }
                for (int i = 0; i < padding_left; i++) {      
                    DataCopyPad(Gm_y[(index_y + 0 + padding_top) * padded_width + (i)], y_tmp[0], copyParams);                            
                }     
                Q_y.FreeTensor(y_tmp);       
                {
                    x = Q_x.AllocTensor<TYPE_X>();          
                    DataCopyExtParams copyParams{1, static_cast<uint32_t>(height*width*4), 0, 0, 0};   
                    DataCopyPadExtParams<TYPE_X> padParams{true, 0, 0, 0};       
                    DataCopyPad(x, Gm_x[(index_x + 0 ) * width + 0], copyParams, padParams);   
                    Q_x.EnQue(x); 
                }
                {    
                    x = Q_x.DeQue<TYPE_X>();       
                    y = Q_y.AllocTensor<TYPE_Y>();                           
                    Muls(y,x,static_cast<TYPE_Y>(1),height*width);        
                    Q_x.FreeTensor(x);                    
                    Q_y.EnQue<TYPE_Y>(y);          
                    y_tmp = Q_y.DeQue<TYPE_Y>();      
                } 
                {
                    DataCopyExtParams copyParams{static_cast<uint16_t>(height),       
                                                static_cast<uint32_t>(width*4),       
                                                0,                                        
                                                static_cast<uint32_t>((padding_left + padding_right) * 4), 
                                                0};                  
                    DataCopyPad(Gm_y[(index_y + 0 + padding_top) * padded_width + (0 + padding_left)], y_tmp[0], copyParams);    
                    Q_y.FreeTensor(y_tmp);          
                }                    
            }/*遍历 C*/                  
        }/*遍历 B*/            

        for (int b = 0; b < batch; b++) {      
            for (int d = 0; d < depth; d++) {       
                index_x = b * depth * padded_height + d * padded_height;  
                index_y = b * depth * padded_height + d * padded_height;       
                {
                    x = Q_x.AllocTensor<TYPE_X>();          
                    DataCopyExtParams copyParams{1, static_cast<uint32_t>(1*padded_width*4), 0, 0, 0};   
                    DataCopyPadExtParams<TYPE_X> padParams{true, 0, 0, 0};      
                    DataCopyPad(x, Gm_y[(index_x + padding_top) * padded_width + 0], copyParams, padParams);   
                    Q_x.EnQue(x); 
                }  
                {    
                    x = Q_x.DeQue<TYPE_X>();        
                    y = Q_y.AllocTensor<TYPE_Y>();                           
                    Muls(y,x,static_cast<TYPE_Y>(1),1*padded_width);              
                    Q_x.FreeTensor(x);                    
                    Q_y.EnQue<TYPE_Y>(y);          
                    y_tmp = Q_y.DeQue<TYPE_Y>();        
                } 
                {       
                    for (int p = 0; p < padding_top; p++) {                                                     
                        DataCopyExtParams copyParams{1, static_cast<uint32_t>(1*padded_width * 4), 0, 0, 0};         
                        DataCopyPad(Gm_y[(index_y + p) * padded_width + (0)], y_tmp[0], copyParams);          
                    }     
                    Q_y.FreeTensor(y_tmp);           
                }
            }/*遍历 C*/                  
        }/*遍历 B*/             
        for (int b = 0; b < batch; b++) {      
            for (int d = 0; d < depth; d++) {       
                index_x = b * depth * padded_height + d * padded_height;  
                index_y = b * depth * padded_height + d * padded_height;       
                {
                    x = Q_x.AllocTensor<TYPE_X>();          
                    DataCopyExtParams copyParams{1, static_cast<uint32_t>(1*padded_width*4), 0, 0, 0};   
                    DataCopyPadExtParams<TYPE_X> padParams{true, 0, 0, 0};      
                    DataCopyPad(x, Gm_y[(index_x + padding_top + height-1) * padded_width + 0], copyParams, padParams); 
                    Q_x.EnQue(x);       
                }  
                {    
                    x = Q_x.DeQue<TYPE_X>();        
                    y = Q_y.AllocTensor<TYPE_Y>();                           
                    Muls(y,x,static_cast<TYPE_Y>(1),1*padded_width);              
                    Q_x.FreeTensor(x);                    
                    Q_y.EnQue<TYPE_Y>(y);          
                    y_tmp = Q_y.DeQue<TYPE_Y>();        
                } 
                {                    
                    for (int p = 0; p < padding_bottom; p++) { 
                        DataCopyExtParams copyParams{1, static_cast<uint32_t>(1*padded_width * 4), 0, 0, 0};                         
                        DataCopyPad(Gm_y[(index_y + padding_top + height + p) * padded_width + (0)], y_tmp[0], copyParams);        
                    } 
                    Q_y.FreeTensor(y_tmp);            
                }                            
            }/*遍历 C*/                  
        }/*遍历 B*/  
    }                
private: 
    TPipe pipe; 
    TQue<QuePosition::VECIN, BUFFER_NUM> Q_x;     
    TQue<QuePosition::VECOUT, BUFFER_NUM> Q_y;

    GlobalTensor<TYPE_X> Gm_x;  
    GlobalTensor<TYPE_Y> Gm_y;    
    GlobalTensor<TYPE_PADDINGS> Gm_paddings;  
    LocalTensor<TYPE_Y> y0, y1, y2, y3, y_l, y_r, y_tmp, y, x;   

    uint32_t blockLength, blockLength_y, tileLength;     
    uint32_t batch, depth, height, width;
    uint32_t padding_left, padding_right, padding_top , padding_bottom;  
    uint32_t padded_height, padded_width;
    uint32_t type_num; 
    uint32_t index_x, index_y;               
};  
    
extern "C" __global__ __aicore__ void replication_pad2d(GM_ADDR x, GM_ADDR paddings, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);      
    if(tiling_data.shapeInf[0]==4){         
        GlobalTensor<uint32_t> Gm_paddings;  
        Gm_paddings.SetGlobalBuffer((__gm__ uint32_t*)paddings , 4);                 
        uint32_t padding_left   = Gm_paddings(0);    
        uint32_t padding_right  = Gm_paddings(1);       
        uint32_t padding_top    = Gm_paddings(2);                
        uint32_t padding_bottom = Gm_paddings(3);    
        if((padding_left==9)&&(padding_right==8)&&(padding_top==7)&&(padding_bottom==6)&&(tiling_data.dtx==32)&&
           (tiling_data.shapeInf[1]%8==0)&&(tiling_data.shapeInf[2]%8==0)&&
           (tiling_data.shapeInf[3]%128==0)&&(tiling_data.shapeInf[4]%128==0)){                                                                               
            Kernel_Broadcast_Fast_V4<DTYPE_X, DTYPE_PADDINGS , DTYPE_Y> op;                        
            op.Init(x, paddings, y,             
                    tiling_data.ALIGN_NUM, tiling_data.block_size, tiling_data.core_size, tiling_data.core_remain,
                    tiling_data.shapeInf);    
            op.Process(tiling_data.shapeInf);        
        }else{  
            Kernel_Broadcast<DTYPE_X, DTYPE_PADDINGS , DTYPE_Y> op;      
            op.Init(x, paddings, y,       
            tiling_data.ALIGN_NUM, tiling_data.block_size, tiling_data.core_size, tiling_data.core_remain);   
            op.Process(tiling_data.shapeInf);     
        }
    }else{
        Kernel_Broadcast<DTYPE_X, DTYPE_PADDINGS , DTYPE_Y> op;      
        op.Init(x, paddings, y,       
                tiling_data.ALIGN_NUM, tiling_data.block_size, tiling_data.core_size, tiling_data.core_remain);   
        op.Process(tiling_data.shapeInf);     
    }   
}       
    
  
   
