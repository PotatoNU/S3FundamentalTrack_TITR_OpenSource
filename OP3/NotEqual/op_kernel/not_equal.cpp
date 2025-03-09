#include "kernel_operator.h"
#include <type_traits>
using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2; 
template<typename TYPE_X1, typename TYPE_X2, typename TYPE_Y> class KernelNotEqual {
    using T = TYPE_X1;  
public:   
    __aicore__ inline KernelNotEqual() {}       
    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, uint8_t ALIGN_NUM, uint32_t block_size, uint32_t core_size, uint32_t core_remain){
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
        pipe.InitBuffer(B_bits, this->tileLength * sizeof(uint8_t));   
        pipe.InitBuffer(B_result, this->tileLength * sizeof(half));  
        pipe.InitBuffer(B_zero, this->tileLength * sizeof(half));
        this->zero = B_zero.Get<half>();
        Duplicate(this->zero, half(0), this->tileLength);  
        if constexpr (std::is_same_v<T, int32_t>) {
            pipe.InitBuffer(B_x1, this->tileLength * sizeof(float));
            pipe.InitBuffer(B_x2, this->tileLength * sizeof(float));
            auto x2 = B_x2.Get<float>();
            Duplicate(x2, float(0), this->tileLength);
        }
        else if constexpr (std::is_same_v<T, float>) {
            pipe.InitBuffer(B_x2, this->tileLength * sizeof(float));
            auto x2 = B_x2.Get<float>();
            Duplicate(x2, float(0), this->tileLength);
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
        CopyOut(loopCount - 1, (length + 31) / 32 * 32);
    }
private:
    __aicore__ inline void CopyIn(int32_t progress, uint32_t length) {
        LocalTensor<TYPE_X1> x1 = Q_x1.AllocTensor<TYPE_X1>();
        LocalTensor<TYPE_X2> x2 = Q_x2.AllocTensor<TYPE_X2>();
        DataCopy(x1, Gm_x1[progress * this->tileLength], length);
        DataCopy(x2, Gm_x2[progress * this->tileLength], length);
        Q_x1.EnQue(x1);
        Q_x2.EnQue(x2);
    }
    __aicore__ inline void Compute(int32_t progress, uint32_t length) {
        LocalTensor<TYPE_X1> x1 = Q_x1.DeQue<TYPE_X1>();
        LocalTensor<TYPE_X2> x2 = Q_x2.DeQue<TYPE_X2>();
        LocalTensor<TYPE_Y> y = Q_y.AllocTensor<TYPE_Y>();
        auto bits = B_bits.Get<uint8_t>();
        auto result = B_result.Get<half>();
        auto inty = y.template ReinterpretCast<uint8_t>();
        if constexpr (std::is_same_v<T, int8_t>) {
            auto float_x1 = B_x1.Get<half>();
            auto float_x2 = B_x2.Get<half>();
            Cast(float_x1, x1, RoundMode::CAST_NONE, length);
            Cast(float_x2, x2, RoundMode::CAST_NONE, length);
            Sub(float_x1, float_x1, float_x2, length);
            Compare(bits, float_x1, zero, CMPMODE::EQ, length);
            Select(result, bits, zero, half(1), SELMODE::VSEL_TENSOR_SCALAR_MODE, length);
            Cast(inty, result, RoundMode::CAST_ROUND, length);
        }
        else {  
            Sub(x1, x1, x2, length);  
            if constexpr (std::is_same_v<T, int32_t>) {
                auto val = B_x1.Get<float>();
                auto float_zero = B_x2.Get<float>();
                Cast(val, x1, RoundMode::CAST_NONE, length);   
                Compare(bits, val, float_zero, CMPMODE::EQ, length); 
            } 
            else if constexpr (std::is_same_v<T, float>) {
                auto float_zero = B_x2.Get<float>();
                Compare(bits, x1, float_zero, CMPMODE::EQ, length);
            } 
            else { 
                Compare(bits, x1, zero, CMPMODE::EQ, length); 
            }
            Select(result, bits, zero, half(1), SELMODE::VSEL_TENSOR_SCALAR_MODE, length);
            Cast(inty, result, RoundMode::CAST_ROUND, length);
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
    TBuf<QuePosition::VECCALC> B_result, B_zero, B_bits;
    TBuf<QuePosition::VECCALC> B_x1, B_x2;
    LocalTensor<half> zero;
    GlobalTensor<TYPE_X1> Gm_x1; 
    GlobalTensor<TYPE_X2> Gm_x2;
    GlobalTensor<TYPE_Y> Gm_y;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
};

template<typename TYPE_X1, typename TYPE_X2,  typename TYPE_Y> class KernelNotEqual_Broadcast {
    using T = TYPE_Y;    
public:
    __aicore__ inline KernelNotEqual_Broadcast() {}      
    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y,  
                                uint8_t ALIGN_NUM, uint32_t block_size, uint32_t core_size, uint32_t core_remain) {    
        this->blockLength = core_size + core_remain;  
        this->blockLength = this->blockLength + (this->blockLength % ALIGN_NUM ? ALIGN_NUM - this->blockLength % ALIGN_NUM : 0);  
        Gm_x1.SetGlobalBuffer((__gm__ TYPE_X1*)x1, this->blockLength);
        Gm_x2.SetGlobalBuffer((__gm__ TYPE_X2*)x2, this->blockLength);
        Gm_y.SetGlobalBuffer((__gm__ TYPE_Y*)y, this->blockLength);     
    }    
    __aicore__ inline void Process(uint32_t shapeInf[2*4]) {            
        int max_dim=0;  
        for(int i=0;i<2;i++){    
            if(shapeInf[i*4+0]>max_dim){  
                max_dim = shapeInf[i*4+0]; 
            }   
        }    
        if (max_dim == 1) {   
            int max_index = 0;    
            for (int i = 0; i < 2; i++) {
                if (shapeInf[i * 4 + 1] > max_index) {
                    max_index = shapeInf[i * 4 + 1];
                }       
            } 
            for (int i = 0; i < max_index; i++) {     
                int index_x1_i = (shapeInf[0 * 4 + 1] <= 1) ? 0 : i; 
                int index_x2_i = (shapeInf[1 * 4 + 1] <= 1) ? 0 : i; 
                float x1_value  = static_cast<float>(Gm_x1(index_x1_i));  
                float x2_value  = static_cast<float>(Gm_x2(index_x2_i)); 
                if(x1_value!=x2_value){
                    Gm_y(i) = static_cast<TYPE_Y>(1);   
                }else{
                    Gm_y(i) = static_cast<TYPE_Y>(0);   
                }   
            }     
        }  
        else if (max_dim == 2) {  
            int max_index[2] = {};  
            for (int i = 0; i < 2; i++) { 
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
                    float x1_value  = static_cast<float>(Gm_x1 (index_x1_i  * shapeInf[0 * 4 + 2] + index_x1_j));   
                    float x2_value  = static_cast<float>(Gm_x2 (index_x2_i  * shapeInf[1 * 4 + 2] + index_x2_j));  
                    if(x1_value!=x2_value){
                        Gm_y(i * max_index[1] + j) = static_cast<TYPE_Y>(1);   
                    }else{
                        Gm_y(i * max_index[1] + j) = static_cast<TYPE_Y>(0);    
                    }     
                }   
            }     
        }     
        else if (max_dim == 3){      
            int max_index[3]={}; 
            for(int i=0;i<2;i++){     
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
                        float x1_value  = static_cast<float>(Gm_x1 (index_x1_i  * (shapeInf[0*4+2] * shapeInf[0*4+3]) + index_x1_j  * shapeInf[0*4+3] + index_x1_k));  
                        float x2_value  = static_cast<float>(Gm_x2 (index_x2_i  * (shapeInf[1*4+2] * shapeInf[1*4+3]) + index_x2_j  * shapeInf[1*4+3] + index_x2_k));  // 从 input_end 的 [i, 1, k] 进行广播
                        if(x1_value!=x2_value){
                            Gm_y(i * (max_index[1] * max_index[2]) + j * max_index[2] + k) = static_cast<TYPE_Y>(1);   
                        }else{
                            Gm_y(i * (max_index[1] * max_index[2]) + j * max_index[2] + k) = static_cast<TYPE_Y>(0);   
                        }    
                    }   
                }   
            }             
        }           
    }      
private:  
    GlobalTensor<TYPE_X1> Gm_x1;     
    GlobalTensor<TYPE_X2> Gm_x2;  
    GlobalTensor<TYPE_Y> Gm_y;     
    uint32_t blockLength;   
};
extern "C" __global__ __aicore__ void not_equal(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling); 
    if (tiling_data.boardCast) {           
        KernelNotEqual_Broadcast<DTYPE_X1, DTYPE_X2, DTYPE_Y> op;    
        op.Init(x1, x2, y,      
                tiling_data.ALIGN_NUM, tiling_data.block_size, tiling_data.core_size, tiling_data.core_remain);   
        op.Process(tiling_data.shapeInf);     
    }else {  
        KernelNotEqual<DTYPE_X1, DTYPE_X2, DTYPE_Y> op;   
        op.Init(x1, x2, y, 
                tiling_data.ALIGN_NUM, tiling_data.block_size, tiling_data.core_size, tiling_data.core_remain); 
        op.Process();    
    }   
}       
    
    