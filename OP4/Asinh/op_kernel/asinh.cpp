#include "kernel_operator.h"
#include <type_traits>             
using namespace AscendC;     
constexpr int32_t BUFFER_NUM = 2;     
template<typename T> struct Map {using type = T;};      
template<> struct Map<int8_t> {using type = half;};             
template<typename TYPE_X,  typename TYPE_Y> class KernelAsinh_Fast {    
    using T = TYPE_Y;           
public:                            
    __aicore__ inline KernelAsinh_Fast() {}      
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t block_size, uint32_t core_size, uint32_t core_remain, bool isTs) {                        
        this->blockLength = core_size + core_remain;  
        this->tileLength = block_size;                                                           
        if constexpr (std::is_same_v<T, half>){
            this->blockLength = this->blockLength + (this->blockLength % 16 ? 16 - this->blockLength % 16 : 0); 
        }else{
            this->blockLength = this->blockLength + (this->blockLength % 8 ? 8 - this->blockLength % 8 : 0);
        }         
        Gm_x.SetGlobalBuffer((__gm__ TYPE_X*)x , this->blockLength);   
        Gm_y.SetGlobalBuffer((__gm__ TYPE_Y*)y , this->blockLength); 
        this->tileNum = this->blockLength / this->tileLength + (this->blockLength % this->tileLength > 0);
        pipe.InitBuffer(Q_x, BUFFER_NUM, this->tileLength * sizeof(TYPE_X)); 
        pipe.InitBuffer(Q_y, BUFFER_NUM, this->tileLength * sizeof(TYPE_Y));   
        this->isTs=isTs;   
        if constexpr (std::is_same_v<T, half>){  
            pipe.InitBuffer(tmpXBuffer, this->tileLength * sizeof(float)); 
            pipe.InitBuffer(tmpYBuffer, this->tileLength * sizeof(float));    
        }   
    }    
    __aicore__ inline void Process() { 
        for (int32_t i = 0; i < this->tileNum-1; i++) {       
            CopyIn(i, this->tileLength);
            Compute(i, this->tileLength);  
            CopyOut(i, this->tileLength);  
        }
        uint32_t length = this->blockLength - this->tileLength * (this->tileNum - 1);
        CopyIn(this->tileNum - 1, length);
        Compute(this->tileNum - 1, length); 
        CopyOut(this->tileNum - 1, length);      
    }

private:
    __aicore__ inline void CopyIn(int32_t progress, uint32_t length) {
        LocalTensor<TYPE_X>  x = Q_x.AllocTensor<TYPE_X>();    
        DataCopy(x, Gm_x[progress * this->tileLength], length);  
        Q_x.EnQue(x);      
    } 
    __aicore__ inline void Compute(int32_t progress, uint32_t length) { 
        LocalTensor<TYPE_X> x = Q_x.DeQue<TYPE_X>();
        LocalTensor<TYPE_Y> y = Q_y.AllocTensor<TYPE_Y>();     
        if(this->isTs){ 
            Mul(y, x, x, length);
            Adds(y, y, static_cast<TYPE_Y>(1), length);       
            Sqrt(y, y, length); 
            Add(y, y, x, length);
            Ln(y, y, length);          
        }else{        
            if constexpr (std::is_same_v<T, half>) {      
                LocalTensor<float> tmpX  = tmpXBuffer.Get<float>();  
                LocalTensor<float> tmpY  = tmpYBuffer.Get<float>();      
                Cast(tmpX, x, RoundMode::CAST_NONE, length);    
                Mul(tmpY, tmpX, tmpX, length);                     // x^2  
                Adds(tmpY, tmpY, static_cast<float>(1), length);   // x*x+1           
                Sqrt(tmpY, tmpY, length);                          // sqrt(x*x+1)    
                Sub(tmpY, tmpY, tmpX, length);                     // sqrt(x*x+1)-x
                Ln(tmpY, tmpY, length);                            // ln[sqrt(x*x+1)-x]
                Muls(tmpY, tmpY, static_cast<float>(-1), length);  // -ln[sqrt(x*x+1)-x]                 
                Cast(y, tmpY, RoundMode::CAST_ROUND, length);               
            }else{      
                Mul(y, x, x, length);                        // x^2      
                Adds(y, y, static_cast<TYPE_Y>(1), length);  // x*x+1    
                Sqrt(y, y, length);                          // sqrt(x*x+1)
                Sub(y, y, x, length);                        // sqrt(x*x+1)-x  
                Ln(y, y, length);                            // ln[sqrt(x*x+1)-x]
                Muls(y, y, static_cast<TYPE_Y>(-1), length); // -ln[sqrt(x*x+1)-x] 
            }      
        } 
        Q_x.FreeTensor(x);  
        Q_y.EnQue<TYPE_Y>(y); 
    }    
    __aicore__ inline void CopyOut(int32_t progress, uint32_t length) {
        LocalTensor<TYPE_Y> y = Q_y.DeQue<TYPE_Y>();
        DataCopy(Gm_y[progress * this->tileLength], y, length);
        Q_y.FreeTensor(y);
    } 
    
private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> Q_x;    
    TQue<QuePosition::VECOUT, BUFFER_NUM> Q_y;
    TBuf<QuePosition::VECCALC> tmpXBuffer,tmpYBuffer;   
    GlobalTensor<TYPE_X> Gm_x; 
    GlobalTensor<TYPE_Y> Gm_y;  
    uint32_t blockLength;   
    uint32_t tileNum; 
    uint32_t tileLength;     
    bool isTs; 
};      
    
extern "C" __global__ __aicore__ void asinh(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);     
    KernelAsinh_Fast<DTYPE_X, DTYPE_Y> op;  
    op.Init(x, y,     
            tiling_data.block_size, tiling_data.core_size, tiling_data.core_remain,tiling_data.isTs);   
    op.Process();       
}  
  
   
   
      