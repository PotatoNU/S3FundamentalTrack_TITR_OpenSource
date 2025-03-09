#define K_MAX_SHAPE_DIM 0 
#include "kernel_operator.h" 
using namespace AscendC; 
constexpr int32_t BUFFER_NUM = 2;  
template<typename T> struct Map {using type = T;};      
template<> struct Map<int8_t> {using type = half;};            

template<typename TYPE_X1, typename TYPE_X2, typename TYPE_Y> class KernelIsClose_Broadcast {
    using T = TYPE_X1;      
public: 
    __aicore__ inline KernelIsClose_Broadcast() {}      
    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y,  
                                uint8_t ALIGN_NUM, uint32_t block_size, uint32_t core_size, uint32_t core_remain,
                                float rtol, float atol) {    
        this->blockLength = core_size + core_remain;  
        this->blockLength = this->blockLength + (this->blockLength % ALIGN_NUM ? ALIGN_NUM - this->blockLength % ALIGN_NUM : 0); 
        this->blockLength = this->blockLength;   
        this->rtol = rtol; 
        this->atol = atol;  
        Gm_x1.SetGlobalBuffer((__gm__ TYPE_X1*)x1, this->blockLength);
        Gm_x2.SetGlobalBuffer((__gm__ TYPE_X2*)x2, this->blockLength); 
        Gm_y.SetGlobalBuffer((__gm__ TYPE_Y*)y, this->blockLength);     
    }      
    __aicore__ inline void Process(uint32_t shapeInf[10]) {     
        int max_dim=0;  
        for(int i=0;i<2;i++){    
            if(shapeInf[i*5+0]>max_dim){  
                max_dim = shapeInf[i*5+0]; 
            }      
        }    
        uint32_t x1_bits=0;   
        uint32_t x2_bits=0; 
        bool x1_is_nan = false; 
        bool x2_is_nan = false; 
        if (max_dim == 1) {   
            int max_index = 0;    
            for (int i = 0; i < 2; i++) {
                if (shapeInf[i * 5 + 1] > max_index) {
                    max_index = shapeInf[i * 5 + 1];
                }        
            } 
            for (int i = 0; i < max_index; i++) {     
                int index_x1_i = (shapeInf[0 * 5 + 1] <= 1) ? 0 : i;  
                int index_x2_i = (shapeInf[1 * 5 + 1] <= 1) ? 0 : i;  
                float x1_value = 0;  
                float x2_value = 0;    
                if constexpr (std::is_same_v<T, uint8_t>) { 
                    int x1_8 = static_cast<int>(Gm_x1(index_x1_i));   
                    int x2_8 = static_cast<int>(Gm_x2(index_x2_i));  
                    x1_value  = static_cast<float>(x1_8);  
                    x2_value  = static_cast<float>(x2_8);    
                }else{ 
                    x1_value  = static_cast<float>(Gm_x1(index_x1_i));  
                    x2_value  = static_cast<float>(Gm_x2(index_x2_i));   
                }    
                if (x1_is_nan && x2_is_nan) {
                    Gm_y(i) = static_cast<TYPE_Y>(1);
                }else if ((x1_is_nan && !x2_is_nan) || (!x1_is_nan && x2_is_nan)) {
                    Gm_y(i) = static_cast<TYPE_Y>(0);
                }else{    
                    x1_value = x1_value-x2_value;
                    x1_value = (x1_value >= 0) ? x1_value : -x1_value;  
                    x2_value = (x2_value >= 0) ? x2_value : -x2_value; 
                    x2_value = x2_value*this->rtol;
                    x2_value = x2_value+this->atol;
                    if(x1_value<=x2_value){
                        Gm_y(i) = static_cast<TYPE_Y>(1);   
                    }else{
                        Gm_y(i) = static_cast<TYPE_Y>(0);   
                    }   
                }
            }     
        }  
        else if (max_dim == 2) {  
            int max_index[2] = {};  
            for (int i = 0; i < 2; i++) { 
                for (int j = 1; j <= shapeInf[i * 5 + 0]; j++) {  
                    if (shapeInf[i * 5 + j] > max_index[j - 1]) {
                        max_index[j - 1] = shapeInf[i * 5 + j];
                    }
                }  
            } 
            for (int i = 0; i < max_index[0]; i++) {           
                for (int j = 0; j < max_index[1]; j++) {   
                    int index_x1_i = (shapeInf[0 * 5 + 1] <= 1) ? 0 : i;  
                    int index_x1_j = (shapeInf[0 * 5 + 2] <= 1) ? 0 : j; 
                    int index_x2_i = (shapeInf[1 * 5 + 1] <= 1) ? 0 : i;    
                    int index_x2_j = (shapeInf[1 * 5 + 2] <= 1) ? 0 : j;   
                    float x1_value = 0;  
                    float x2_value = 0;   
                    if constexpr (std::is_same_v<T, uint8_t>) { 
                        int x1_8 = static_cast<int>(Gm_x1 (index_x1_i  * shapeInf[0 * 5 + 2] + index_x1_j));  
                        int x2_8 = static_cast<int>(Gm_x2 (index_x2_i  * shapeInf[1 * 5 + 2] + index_x2_j));  
                        x1_value  = static_cast<float>(x1_8);  
                        x2_value  = static_cast<float>(x2_8);    
                    }else{ 
                        x1_value  = static_cast<float>(Gm_x1 (index_x1_i  * shapeInf[0 * 5 + 2] + index_x1_j));  
                        x2_value  = static_cast<float>(Gm_x2 (index_x2_i  * shapeInf[1 * 5 + 2] + index_x2_j));   
                    }  
                    if (x1_is_nan && x2_is_nan) {
                        Gm_y(i * max_index[1] + j) = static_cast<TYPE_Y>(1);
                    }else if ((x1_is_nan && !x2_is_nan) || (!x1_is_nan && x2_is_nan)) {
                        Gm_y(i * max_index[1] + j) = static_cast<TYPE_Y>(0);
                    }else{     
                        x1_value = x1_value-x2_value;
                        x1_value = (x1_value >= 0) ? x1_value : -x1_value;  

                        x2_value = (x2_value >= 0) ? x2_value : -x2_value; 
                        x2_value = x2_value*this->rtol;
                        x2_value = x2_value+this->atol;
                
                        if(x1_value<=x2_value){
                            Gm_y(i * max_index[1] + j) = static_cast<TYPE_Y>(1);   
                        }else{
                            Gm_y(i * max_index[1] + j) = static_cast<TYPE_Y>(0);    
                        }    
                    }
                }   
            }     
        }     
        else if (max_dim == 3){      
            int max_index[3]={}; 
            for(int i=0;i<2;i++){     
                for (int j = 1; j <= shapeInf[i*5+0]; j++) {  
                    if(shapeInf[i*5+j]>max_index[j-1]){
                        max_index[j-1] = shapeInf[i*5+j];
                    } 
                }     
            }  
            for (int i = 0; i < max_index[0]; i++) {            
                for (int j = 0; j < max_index[1]; j++) {   
                    for (int k = 0; k < max_index[2]; k++) {          
                        int index_x1_i = (shapeInf[0 * 5 + 1] <= 1) ? 0 : i;
                        int index_x1_j = (shapeInf[0 * 5 + 2] <= 1) ? 0 : j;
                        int index_x1_k = (shapeInf[0 * 5 + 3] <= 1) ? 0 : k;  
                        int index_x2_i = (shapeInf[1 * 5 + 1] <= 1) ? 0 : i; 
                        int index_x2_j = (shapeInf[1 * 5 + 2] <= 1) ? 0 : j; 
                        int index_x2_k = (shapeInf[1 * 5 + 3] <= 1) ? 0 : k; 
                        float x1_value = 0;  
                        float x2_value = 0;   
                        if constexpr (std::is_same_v<T, uint8_t>) { 
                            int x1_8 = static_cast<int>(Gm_x1 (index_x1_i  * (shapeInf[0*5+2] * shapeInf[0*5+3]) + index_x1_j  * shapeInf[0*5+3] + index_x1_k));  
                            int x2_8 = static_cast<int>(Gm_x2 (index_x2_i  * (shapeInf[1*5+2] * shapeInf[1*5+3]) + index_x2_j  * shapeInf[1*5+3] + index_x2_k));  
                            x1_value  = static_cast<float>(x1_8);  
                            x2_value  = static_cast<float>(x2_8);    
                        }else{ 
                            x1_value  = static_cast<float>(Gm_x1 (index_x1_i  * (shapeInf[0*5+2] * shapeInf[0*5+3]) + index_x1_j  * shapeInf[0*5+3] + index_x1_k));  
                            x2_value  = static_cast<float>(Gm_x2 (index_x2_i  * (shapeInf[1*5+2] * shapeInf[1*5+3]) + index_x2_j  * shapeInf[1*5+3] + index_x2_k));   
                        }  
                        if (x1_is_nan && x2_is_nan) {   
                            Gm_y(i * (max_index[1] * max_index[2]) + j * max_index[2] + k) = static_cast<TYPE_Y>(1);  
                        }else if ((x1_is_nan && !x2_is_nan) || (!x1_is_nan && x2_is_nan)) {
                            Gm_y(i * (max_index[1] * max_index[2]) + j * max_index[2] + k) = static_cast<TYPE_Y>(0);
                        }
                        else{        
                            x1_value = x1_value-x2_value;
                            x1_value = (x1_value >= 0) ? x1_value : -x1_value;  
                            x2_value = (x2_value >= 0) ? x2_value : -x2_value; 
                            x2_value = x2_value*this->rtol;
                            x2_value = x2_value+this->atol;  
                            if(x1_value<=x2_value){
                                Gm_y(i * (max_index[1] * max_index[2]) + j * max_index[2] + k) = static_cast<TYPE_Y>(1);   
                            }else{
                                Gm_y(i * (max_index[1] * max_index[2]) + j * max_index[2] + k) = static_cast<TYPE_Y>(0);   
                            }    
                        } 
                    }   
                }   
            }              
        }               
        else if (max_dim == 4){              
            int max_index[4] = {}; 
            int index_x1_i, index_x1_j, index_x1_k, index_x1_l;
            int index_x2_i, index_x2_j, index_x2_k, index_x2_l;
            float x1_value = 0;          
            float x2_value = 0;                
            int x1_8,x2_8;
            int index1,index2,index3;  
            for (int i = 1; i < 2; i++) { 
                for (int j = 1; j <= shapeInf[i * 5 + 0]; j++) {
                    if (shapeInf[i * 5 + j] > max_index[j - 1]) {
                        max_index[j - 1] = shapeInf[i * 5 + j];
                    } 
                } 
            } 
            for (int i = 0; i < max_index[0]; i++) {  
                for (int j = 0; j < max_index[1]; j++) { 
                    for (int k = 0; k < max_index[2]; k++) {
                        for (int l = 0; l < max_index[3]; l++) {
                            index_x1_i = 0;
                            index_x1_j = 0;
                            index_x1_k = 0; 
                            index_x1_l = (shapeInf[0 * 5 + 1] <= 1) ? 0 : l; 
                            index_x2_i = (shapeInf[1 * 5 + 1] <= 1) ? 0 : i;  
                            index_x2_j = (shapeInf[1 * 5 + 2] <= 1) ? 0 : j;       
                            index_x2_k = (shapeInf[1 * 5 + 3] <= 1) ? 0 : k;  
                            index_x2_l = (shapeInf[1 * 5 + 4] <= 1) ? 0 : l;     
                            if constexpr (std::is_same_v<T, uint8_t>) {  
                                index1 = index_x1_i * (shapeInf[0 * 5 + 2] * shapeInf[0 * 5 + 3] * shapeInf[0 * 5 + 4]) +
                                         index_x1_j * (shapeInf[0 * 5 + 3] * shapeInf[0 * 5 + 4]) +
                                         index_x1_k * shapeInf[0 * 5 + 4] + index_x1_l;  
                                index2 = index_x2_i * (shapeInf[1 * 5 + 2] * shapeInf[1 * 5 + 3] * shapeInf[1 * 5 + 4]) +
                                         index_x2_j * (shapeInf[1 * 5 + 3] * shapeInf[1 * 5 + 4]) +
                                         index_x2_k * shapeInf[1 * 5 + 4] + index_x2_l; 
                                x1_8 = static_cast<int>(Gm_x1(index1));  
                                x2_8 = static_cast<int>(Gm_x2(index2));        
                                x1_value  = static_cast<float>(x1_8);     
                                x2_value  = static_cast<float>(x2_8);      
                            }    
                            x1_value = x1_value - x2_value;
                            x1_value = (x1_value < 0) ? -x1_value : x1_value;             
                            x2_value = (x2_value < 0) ? -x2_value : x2_value;       
                            x2_value = x2_value * this->rtol;   
                            x2_value = x2_value + this->atol;    
                            index3 = i * (max_index[1] * max_index[2] * max_index[3]) + j * (max_index[2] * max_index[3]) + k * max_index[3] + l;
                            if (x1_value <= x2_value) {    
                                Gm_y.SetValue(index3,static_cast<TYPE_Y>(1));      
                            } else {        
                                Gm_y.SetValue(index3,static_cast<TYPE_Y>(0));
                            }     
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
    uint32_t blockLength;   
    float rtol;
    float atol;  
};

template<typename TYPE_X1, typename TYPE_X2, typename TYPE_Y> class KernelIsClose_FP32 {
    using T = TYPE_X1;  
public:         
    __aicore__ inline KernelIsClose_FP32(){}         
    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, TPipe* pipeIn, 
                                uint32_t block_size, uint32_t core_size, uint32_t core_remain, 
                                float rtol, float atol) {    
        this->blockLength = core_size + core_remain;
        this->blockLength = this->blockLength + (this->blockLength % 8 ? 8 - this->blockLength % 8 : 0);
        this->tileLength = block_size;   
        Gm_x1.SetGlobalBuffer((__gm__ TYPE_X1*)x1 , this->blockLength); 
        Gm_x2.SetGlobalBuffer((__gm__ TYPE_X2*)x2 , this->blockLength);   
        Gm_y.SetGlobalBuffer((__gm__ TYPE_Y*)y , this->blockLength);
        this->rtol = rtol;  
        this->atol = atol;  
        this->tileNum = this->blockLength / this->tileLength + (this->blockLength % this->tileLength > 0);
        pipe = pipeIn;   
        pipe->InitBuffer(Q_x1, BUFFER_NUM, this->tileLength * 4); 
        pipe->InitBuffer(Q_x2, BUFFER_NUM, this->tileLength * 4);
        pipe->InitBuffer(Q_y, BUFFER_NUM,  this->tileLength * 4); 
        pipe->InitBuffer(B_bits,   this->tileLength * 1); 
        pipe->InitBuffer(B_result, this->tileLength * 2); 
        pipe->InitBuffer(B_zero,   this->tileLength * 2); 
        this->zero = B_zero.Get<half>();  
        Duplicate(this->zero, half(0), this->tileLength);    
    }
    __aicore__ inline void Process() {
        int32_t loopCount = this->tileNum-24;                              
        for (int32_t i = 0; i < loopCount-1; i++) Compute(i, this->tileLength);
        uint32_t length = this->blockLength - this->tileLength * (loopCount - 1);
        Compute(loopCount - 1, length);  
    }   
private: 
    __aicore__ inline void Compute(int32_t progress, uint32_t length) {
        x1 = Q_x1.AllocTensor<TYPE_X1>();
        x2 = Q_x2.AllocTensor<TYPE_X2>();
        DataCopy(x1, Gm_x1[progress * this->tileLength], length);
        DataCopy(x2, Gm_x2[progress * this->tileLength], length);
        Q_x1.EnQue(x1); 
        Q_x2.EnQue(x2);

        x1 = Q_x1.DeQue<TYPE_X1>();
        x2 = Q_x2.DeQue<TYPE_X2>();
        y = Q_y.AllocTensor<TYPE_Y>(); 
        auto bits   = B_bits.Get<uint8_t>();
        auto result = B_result.Get<half>();
        auto inty   = y.template ReinterpretCast<uint8_t>();
        if constexpr (std::is_same_v<T, float>) {
            Sub(x1, x1, x2, length);    
            Abs(x1, x1, length);                     
            Abs(x2, x2, length);       
            Muls(x2, x2, this->rtol, length);                    
            Adds(x2, x2, this->atol, length);                    
            Compare(bits, x1, x2, CMPMODE::GT, length);    
            Select(result, bits, zero, half(1), SELMODE::VSEL_TENSOR_SCALAR_MODE, length);
            Cast(inty, result, RoundMode::CAST_ROUND, length);  
        }    
        Q_x1.FreeTensor(x1);
        Q_x2.FreeTensor(x2);
        Q_y.EnQue<TYPE_Y>(y);
        y = Q_y.DeQue<TYPE_Y>(); 
        DataCopy(Gm_y[progress * this->tileLength], y, length);
        Q_y.FreeTensor(y);  
    }

private:
    TPipe* pipe;        
    TQue<QuePosition::VECIN, BUFFER_NUM> Q_x1, Q_x2;
    TQue<QuePosition::VECOUT, BUFFER_NUM> Q_y;
    TBuf<QuePosition::VECCALC> B_result, B_zero, B_bits; 
    LocalTensor<half> zero;
    LocalTensor<TYPE_X1> x1, x2;
    LocalTensor<TYPE_Y> y;
    GlobalTensor<TYPE_X1> Gm_x1, Gm_x2; 
    GlobalTensor<TYPE_Y> Gm_y; 
    uint32_t blockLength, tileNum, tileLength;  
    float rtol, atol;
}; 

extern "C" __global__ __aicore__ void is_close(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);       
    if(TILING_KEY_IS(1)){          
        TPipe pipe;                    
        KernelIsClose_FP32<DTYPE_X1, DTYPE_X2, DTYPE_Y> op;    
        op.Init(x1, x2, y, &pipe,    
                tiling_data.block_size, tiling_data.core_size, tiling_data.core_remain,
                tiling_data.rtol, tiling_data.atol);     
        op.Process();        
    }else if(TILING_KEY_IS(0)){             
        KernelIsClose_Broadcast<DTYPE_X1, DTYPE_X2, DTYPE_Y> op;     
        op.Init(x1, x2, y,      
                tiling_data.ALIGN_NUM, tiling_data.block_size, tiling_data.core_size, tiling_data.core_remain,
                tiling_data.rtol, tiling_data.atol);    
        op.Process(tiling_data.shapeInf);                                                       
    }               
}                
    