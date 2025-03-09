#define K_MAX_SHAPE_DIM 0     
#include "kernel_operator.h"
using namespace AscendC;     
constexpr int32_t BUFFER_NUM = 2;           
template<typename T> struct Map {using type = T;};                 
template<> struct Map<int8_t> {using type = half;};             
template<typename TYPE_X1, typename TYPE_X2, typename TYPE_Y> class KernelDiv_Fast {     
    using T = TYPE_Y;         
public:                            
    __aicore__ inline KernelDiv_Fast() {}  
    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2,GM_ADDR y, 
                                uint32_t ALIGN_NUM, uint32_t block_size, uint32_t core_size, uint32_t core_remain) {                      
        this->blockLength = core_size + core_remain;  
        this->tileLength = block_size;                                                            
        this->blockLength = this->blockLength + (this->blockLength % ALIGN_NUM ? ALIGN_NUM - this->blockLength % ALIGN_NUM : 0); 
        Gm_x1.SetGlobalBuffer((__gm__ TYPE_X1*)x1 , this->blockLength);
        Gm_x2.SetGlobalBuffer((__gm__ TYPE_X2*)x2 , this->blockLength);
        Gm_y.SetGlobalBuffer((__gm__ TYPE_Y*)y , this->blockLength); 
        this->tileNum = this->blockLength / this->tileLength + (this->blockLength % this->tileLength > 0);
        pipe.InitBuffer(Q_x1, BUFFER_NUM, this->tileLength * sizeof(TYPE_X1)); 
        pipe.InitBuffer(Q_x2, BUFFER_NUM, this->tileLength * sizeof(TYPE_X2));  
        pipe.InitBuffer(Q_y, BUFFER_NUM, this->tileLength * sizeof(TYPE_Y));   
        if constexpr (std::is_same_v<T, int32_t>) {
            pipe.InitBuffer(B_x1, this->tileLength * sizeof(float));   
            pipe.InitBuffer(B_x2, this->tileLength * sizeof(float));
        }  
        else if constexpr (std::is_same_v<T, int8_t>) {  
            pipe.InitBuffer(B_x1, this->tileLength * sizeof(half));
            pipe.InitBuffer(B_x2, this->tileLength * sizeof(half));
        } 
    }    
    __aicore__ inline void Process() { 
        int32_t loopCount = this->tileNum;              
        for (int32_t i = 0; i < loopCount-1; i++) {   
            CopyIn(i, this->tileLength);
            Compute(i, this->tileLength);  
            CopyOut(i, this->tileLength);
        }
        uint32_t length = this->blockLength - this->tileLength * (loopCount - 1);
        CopyIn(loopCount - 1, length);
        Compute(loopCount - 1, length); 
        CopyOut(loopCount - 1, length);    
    }
          
private:
    __aicore__ inline void CopyIn(int32_t progress, uint32_t length) {
        LocalTensor<TYPE_X1>  x1 = Q_x1.AllocTensor<TYPE_X1>();    
        LocalTensor<TYPE_X2>  x2 = Q_x2.AllocTensor<TYPE_X2>();   
        DataCopy(x1, Gm_x1[progress * this->tileLength], length); 
        DataCopy(x2, Gm_x2[progress * this->tileLength], length); 
        Q_x1.EnQue(x1);      
        Q_x2.EnQue(x2);        
    } 
    __aicore__ inline void Compute(int32_t progress, uint32_t length) { 
        LocalTensor<TYPE_X1> x1 = Q_x1.DeQue<TYPE_X1>();
        LocalTensor<TYPE_X2> x2 = Q_x2.DeQue<TYPE_X2>();  
        LocalTensor<TYPE_Y> y = Q_y.AllocTensor<TYPE_Y>();     
        if constexpr (std::is_same_v<T, int8_t>) { 
            auto float_x1 = B_x1.Get<half>();   
            auto float_x2 = B_x2.Get<half>();        
            Cast(float_x1, x1, RoundMode::CAST_NONE, length);
            Cast(float_x2, x2, RoundMode::CAST_NONE, length); 
            Div(float_x1, float_x1, float_x2, length);         
            Cast(y, float_x1, RoundMode::CAST_FLOOR, length);    
        }else if constexpr (std::is_same_v<T, int32_t>){
            auto float_x1 = B_x1.Get<float>();     
            auto float_x2 = B_x2.Get<float>();      
            Cast(float_x1, x1, RoundMode::CAST_NONE, length); 
            Cast(float_x2, x2, RoundMode::CAST_NONE, length);   
            Div(float_x1, float_x1, float_x2, length);          
            Cast(y, float_x1, RoundMode::CAST_FLOOR, length); 
        }else{     
            Div(y, x1, x2, length);         
        }       
        Q_x1.FreeTensor(x1); 
        Q_x2.FreeTensor(x2); 
        Q_y.EnQue<TYPE_Y>(y); 
    }   
    __aicore__ inline void CopyOut(int32_t progress, uint32_t length) {
        LocalTensor<TYPE_Y> y = Q_y.DeQue<TYPE_Y>();
        DataCopy(Gm_y[progress * this->tileLength], y, length);
        Q_y.FreeTensor(y);
    } 
private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> Q_x1, Q_x2; 
    TQue<QuePosition::VECOUT, BUFFER_NUM> Q_y;
    TBuf<QuePosition::VECCALC> B_x1, B_x2;
    GlobalTensor<TYPE_X1> Gm_x1; 
    GlobalTensor<TYPE_X2> Gm_x2;  
    GlobalTensor<TYPE_Y> Gm_y; 
    uint32_t blockLength;  
    uint32_t tileNum;
    uint32_t tileLength;    
};    
     
template<typename TYPE_X1, typename TYPE_X2,  typename TYPE_Y> class KernelDiv_Broadcast {
    using T = TYPE_Y;    
public:
    __aicore__ inline KernelDiv_Broadcast() {}     
    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2,GM_ADDR y,  
                                uint32_t ALIGN_NUM, uint32_t block_size, uint32_t core_size, uint32_t core_remain) {    
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!"); 
        this->blockLength = core_size + (GetBlockNum() == GetBlockIdx() + 1 ? core_remain : 0);  
        this->blockLength = this->blockLength + (this->blockLength % ALIGN_NUM ? ALIGN_NUM - this->blockLength % ALIGN_NUM : 0); 
        auto startPointer = core_size * GetBlockIdx();  
        auto bufferlength = this->blockLength; 
        Gm_x1.SetGlobalBuffer((__gm__ TYPE_X1*)x1, bufferlength);  
        Gm_x2.SetGlobalBuffer((__gm__ TYPE_X2*)x2, bufferlength);  
        Gm_y.SetGlobalBuffer((__gm__ TYPE_Y*)y, bufferlength);            
        pipe.InitBuffer(tmp1Buffer, 1 * sizeof(DTYPE_Y));   
        pipe.InitBuffer(tmp2Buffer, 1 * sizeof(DTYPE_Y));   
        pipe.InitBuffer(tmp32Buffer, 1 * sizeof(float));    
  
    }    
    __aicore__ inline void Process(uint32_t shapeInf[2*4]) {    
        LocalTensor<TYPE_Y> tmp1 = tmp1Buffer.Get<TYPE_Y>();     
        LocalTensor<TYPE_Y> tmp2 = tmp2Buffer.Get<TYPE_Y>(); 
        uint32_t input_num=2;          
        int max_dim=0;  
        for(int i=0;i<input_num;i++){ 
            if(shapeInf[i*4+0]>max_dim){  
                max_dim = shapeInf[i*4+0]; 
            }   
        }    
        if (max_dim == 1) {   
            int max_index = 0;   
            for (int i = 0; i < input_num; i++) {
                if (shapeInf[i * 4 + 1] > max_index) {
                    max_index = shapeInf[i * 4 + 1];
                }       
            } 
            for (int i = 0; i < max_index; i++) {     
                int index_x1_i = (shapeInf[0 * 4 + 1] <= 1) ? 0 : i;  
                int index_x2_i = (shapeInf[1 * 4 + 1] <= 1) ? 0 : i;  
                if constexpr (std::is_same_v<T, int8_t> || std::is_same_v<T, int32_t>) { 
                    int x1_value  = static_cast<int>(Gm_x1(index_x1_i));  
                    int x2_value  = static_cast<int>(Gm_x2(index_x2_i));  
                    Gm_y(i) = static_cast<TYPE_Y>(x1_value/x2_value);         
                }else{
                    float x1_value  = static_cast<float>(Gm_x1(index_x1_i));  
                    float x2_value  = static_cast<float>(Gm_x2(index_x2_i)); 
                    Duplicate<TYPE_Y>(tmp1, static_cast<TYPE_Y>(x1_value), 1);    
                    Duplicate<TYPE_Y>(tmp2, static_cast<TYPE_Y>(x2_value), 1);    
                    Div(tmp2, tmp1, tmp2, 1);       
                    Gm_y(i) = static_cast<TYPE_Y>(tmp2(0));       
                }

            }     
        } 
        else if (max_dim == 2) {  
            int max_index[2] = {};  
            for (int i = 0; i < input_num; i++) { 
                for (int j = 1; j <= shapeInf[i * 4 + 0]; j++) {  
                    if (shapeInf[i * 4 + j] > max_index[j - 1]) {
                        max_index[j - 1] = shapeInf[i * 4 + j];
                    }
                }  
            } 
            for (int i = 0; i < max_index[0]; i++) {           
                for (int j = 0; j < max_index[1]; j++) {   
                    int index_x1_i = (shapeInf[0 * 4 + 1] <= 1) ? 0 : i; 
                    int index_x1_j = (shapeInf[0 * 4 + 2] <= 1) ? 0 : j; 
                    int index_x2_i = (shapeInf[1 * 4 + 1] <= 1) ? 0 : i; 
                    int index_x2_j = (shapeInf[1 * 4 + 2] <= 1) ? 0 : j;  
                    if constexpr (std::is_same_v<T, int8_t> || std::is_same_v<T, int32_t>) { 
                        int x1_value  = static_cast<int>(Gm_x1 (index_x1_i  * shapeInf[0 * 4 + 2] + index_x1_j));   
                        int x2_value  = static_cast<int>(Gm_x2 (index_x2_i  * shapeInf[1 * 4 + 2] + index_x2_j));  
                        Gm_y(i * max_index[1] + j) = static_cast<TYPE_Y>(x1_value/x2_value);         
                    }else{
                        float x1_value  = static_cast<float>(Gm_x1 (index_x1_i  * shapeInf[0 * 4 + 2] + index_x1_j));   
                        float x2_value  = static_cast<float>(Gm_x2 (index_x2_i  * shapeInf[1 * 4 + 2] + index_x2_j));  
                        Duplicate<TYPE_Y>(tmp1, static_cast<TYPE_Y>(x1_value), 1);    
                        Duplicate<TYPE_Y>(tmp2, static_cast<TYPE_Y>(x2_value), 1);    
                        Div(tmp2, tmp1, tmp2, 1);       
                        Gm_y(i * max_index[1] + j) = static_cast<TYPE_Y>(tmp2(0));     
                    }
                }   
            }     
        }     
        else if (max_dim == 3){    
            int max_index[3]={}; 
            for(int i=0;i<input_num;i++){    
                for (int j = 1; j <= shapeInf[i*4+0]; j++) {  
                    if(shapeInf[i*4+j]>max_index[j-1]){
                        max_index[j-1] = shapeInf[i*4+j];
                    } 
                }     
            }  
            for (int i = 0; i < max_index[0]; i++) {            
                for (int j = 0; j < max_index[1]; j++) {   
                    for (int k = 0; k < max_index[2]; k++) {          
                        int index_x1_i = (shapeInf[0 * 4 + 1] <= 1) ? 0 : i;  
                        int index_x1_j = (shapeInf[0 * 4 + 2] <= 1) ? 0 : j; 
                        int index_x1_k = (shapeInf[0 * 4 + 3] <= 1) ? 0 : k; 
                        int index_x2_i = (shapeInf[1 * 4 + 1] <= 1) ? 0 : i;  
                        int index_x2_j = (shapeInf[1 * 4 + 2] <= 1) ? 0 : j;  
                        int index_x2_k = (shapeInf[1 * 4 + 3] <= 1) ? 0 : k;  
                        if constexpr (std::is_same_v<T, int8_t> || std::is_same_v<T, int32_t>) { 
                            int x1_value  = static_cast<int>(Gm_x1 (index_x1_i  * (shapeInf[0*4+2] * shapeInf[0*4+3]) + index_x1_j  * shapeInf[0*4+3] + index_x1_k));  
                            int x2_value  = static_cast<int>(Gm_x2 (index_x2_i  * (shapeInf[1*4+2] * shapeInf[1*4+3]) + index_x2_j  * shapeInf[1*4+3] + index_x2_k));  
                            Gm_y(i * (max_index[1] * max_index[2]) + j * max_index[2] + k) = static_cast<TYPE_Y>(x1_value/x2_value);         
                        }else{
                            float x1_value  = static_cast<float>(Gm_x1 (index_x1_i  * (shapeInf[0*4+2] * shapeInf[0*4+3]) + index_x1_j  * shapeInf[0*4+3] + index_x1_k));  
                            float x2_value  = static_cast<float>(Gm_x2 (index_x2_i  * (shapeInf[1*4+2] * shapeInf[1*4+3]) + index_x2_j  * shapeInf[1*4+3] + index_x2_k));  
                            Duplicate<TYPE_Y>(tmp1, static_cast<TYPE_Y>(x1_value), 1);    
                            Duplicate<TYPE_Y>(tmp2, static_cast<TYPE_Y>(x2_value), 1);    
                            Div(tmp2, tmp1, tmp2, 1);       
                            Gm_y(i * (max_index[1] * max_index[2]) + j * max_index[2] + k) = static_cast<TYPE_Y>(tmp2(0));     
                        }
                    }   
                }   
            }              
        }     
    }      
private:
    TPipe pipe;    
    GlobalTensor<TYPE_X1> Gm_x1;     
    GlobalTensor<TYPE_X2> Gm_x2;  
    GlobalTensor<TYPE_Y> Gm_y;  
    TBuf<QuePosition::VECCALC> tmp1Buffer,tmp2Buffer,tmp32Buffer; 
    uint32_t blockLength;   
};  

extern "C" __global__ __aicore__ void div(GM_ADDR x1, GM_ADDR x2,GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);    
    if(TILING_KEY_IS(0)){   
        KernelDiv_Fast<DTYPE_X1, DTYPE_X2, DTYPE_Y> op;  
        op.Init(x1, x2, y,
                tiling_data.ALIGN_NUM, tiling_data.block_size, tiling_data.core_size, tiling_data.core_remain);   
        op.Process();     
    }else if(TILING_KEY_IS(1)){   
        KernelDiv_Broadcast<DTYPE_X1, DTYPE_X2, DTYPE_Y> op;    
        op.Init(x1, x2, y,      
                tiling_data.ALIGN_NUM, tiling_data.block_size, tiling_data.core_size, tiling_data.core_remain);   
        op.Process(tiling_data.shapeInf);   
    }
}   

