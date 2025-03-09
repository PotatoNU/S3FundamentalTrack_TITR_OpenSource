#define K_MAX_SHAPE_DIM 0
#include "kernel_operator.h"    
#define CAST_ON  
using namespace AscendC;
constexpr int8_t BUFFER_NUM = 2;                
template<typename T> struct Map {using type = T;};      
template<> struct Map<int8_t> {using type = half;};            
template <typename TYPE_Y, typename TYPE_DY, typename TYPE_Z> 
class KernelAsinhGrad_Fast 
{
    using T = TYPE_Y;           
public:     
    __aicore__ inline KernelAsinhGrad_Fast() {}    
    __aicore__ inline void Init(GM_ADDR y, GM_ADDR dy, GM_ADDR z,
                                uint16_t block_size, uint32_t core_size){       
        this->blockLength = core_size; 
        this->tileLength = block_size;  
        Gm_y.SetGlobalBuffer((__gm__ TYPE_Y *)y, this->blockLength);
        Gm_dy.SetGlobalBuffer((__gm__ TYPE_DY *)dy, this->blockLength);
        Gm_z.SetGlobalBuffer((__gm__ TYPE_Z *)z, this->blockLength);
        this->tileNum = this->blockLength / this->tileLength + (this->blockLength % this->tileLength > 0);
        pipe.InitBuffer(Q_y, BUFFER_NUM, this->tileLength * sizeof(TYPE_Y));
        pipe.InitBuffer(Q_dy, BUFFER_NUM, this->tileLength * sizeof(TYPE_DY));   
        pipe.InitBuffer(Q_z, BUFFER_NUM, this->tileLength * sizeof(TYPE_Z));
        if constexpr (std::is_same_v<T, half>){  
            pipe.InitBuffer(tmpYBuffer, this->tileLength * sizeof(float));  
            pipe.InitBuffer(tmpDYBuffer, this->tileLength * sizeof(float));
        }       
    }   
    __aicore__ inline void Process()   
    {           
        int32_t loopCount = this->tileNum;
        for (int32_t i = 0; i < loopCount - 1; i++){
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
    __aicore__ inline void CopyIn(int32_t progress, uint32_t length)
    {
        LocalTensor<TYPE_Y> y = Q_y.AllocTensor<TYPE_Y>();
        LocalTensor<TYPE_DY> dy = Q_dy.AllocTensor<TYPE_DY>();
        DataCopy(y, Gm_y[progress * this->tileLength], length);
        DataCopy(dy, Gm_dy[progress * this->tileLength], length);
        Q_y.EnQue(y);
        Q_dy.EnQue(dy);
    }   
    __aicore__ inline void Compute(int32_t progress, uint32_t length)  
    {
        LocalTensor<TYPE_Y> y = Q_y.DeQue<TYPE_Y>();   
        LocalTensor<TYPE_DY> dy = Q_dy.DeQue<TYPE_DY>();
        LocalTensor<TYPE_Z> z = Q_z.AllocTensor<TYPE_Z>();   
        if constexpr (std::is_same_v<T, half>){    
            LocalTensor<float> tmpy = tmpYBuffer.Get<float>();        
            LocalTensor<float> tmpdy = tmpDYBuffer.Get<float>();         
            Cast(tmpy, y, RoundMode::CAST_NONE, length);
            Cast(tmpdy, dy, RoundMode::CAST_NONE, length);                          
            Exp(tmpy, tmpy, length);    
            Mul(tmpdy, tmpdy, tmpy, length);    
            Muls(tmpdy, tmpdy, static_cast<float>(2), length);  
            Mul(tmpy, tmpy, tmpy, length);   
            Adds(tmpy, tmpy, static_cast<float>(1), length);  
            Div(tmpdy, tmpdy, tmpy, length); 
            Cast(z, tmpdy, RoundMode::CAST_ROUND, length);    
        }else{     
            Exp(y, y, length);
            Mul(dy, dy, y, length);
            Muls(dy, dy, static_cast<TYPE_Y>(2), length);  
            Mul(y, y, y, length);
            Adds(y, y, static_cast<TYPE_Y>(1), length);  
            Div(z, dy, y, length); 
        }   
        Q_y.FreeTensor(y);
        Q_dy.FreeTensor(dy);  
        Q_z.EnQue<TYPE_Z>(z); 
    }
    __aicore__ inline void CopyOut(int32_t progress, uint32_t length)
    {
        LocalTensor<TYPE_Z> z = Q_z.DeQue<TYPE_Z>();
        DataCopy(Gm_z[progress * this->tileLength], z, length);
        Q_z.FreeTensor(z); 
    }
private:
    TPipe pipe; 
    TQue<QuePosition::VECIN, BUFFER_NUM> Q_y, Q_dy; 
    TQue<QuePosition::VECOUT, BUFFER_NUM> Q_z;
    TBuf<QuePosition::VECCALC> tmpYBuffer, tmpDYBuffer;      
    GlobalTensor<TYPE_Y> Gm_y;  
    GlobalTensor<TYPE_DY> Gm_dy; 
    GlobalTensor<TYPE_Z> Gm_z;     
    uint32_t blockLength;     
    uint32_t tileNum;  
    uint16_t tileLength;  
};
  
// 6行 4变量 原始公式—>两行fp16计算->最快,但官方1/x精度不够 
// 6行 5变量 原始公式—>两行fp16计算 
template <typename TYPE_Y, typename TYPE_DY, typename TYPE_Z>   
class KernelAsinhGrad_Fast_Fast_V7
{                   
    using T = TYPE_Y;                    
public:
    __aicore__ inline KernelAsinhGrad_Fast_Fast_V7() {}      
    __aicore__ inline void Init(GM_ADDR y, GM_ADDR dy, GM_ADDR z,
                                uint32_t core_size, uint16_t block_size, uint16_t tile_num, TPipe* pipeIn){       
        pipe = pipeIn;       
        this->blockLength = core_size;      
        this->tileLength  = block_size;         
        this->tileNum = tile_num;      
        pipe->InitBuffer(Q_y, 2,  this->tileLength * 2);                  
        pipe->InitBuffer(Q_dy,2,  this->tileLength * 2);   
        #ifdef CAST_ON 
        pipe->InitBuffer(tmpYBuffer,    this->tileLength * 4);                            
        pipe->InitBuffer(tmpDYBuffer,   this->tileLength * 4);            
        tmpy  = tmpYBuffer.Get<float>();                       
        tmpdy = tmpDYBuffer.Get<float>();          
        tmp025 = tmp025YBuffer.Get<float>();        
        Duplicate<float>(tmp025, 1.0f, this->tileLength);                                               
        #endif    
        Gm_y.SetGlobalBuffer((__gm__  half *)y,  this->blockLength);   
        Gm_dy.SetGlobalBuffer((__gm__ half *)dy, this->blockLength);                            
        Gm_z.SetGlobalBuffer((__gm__  half *)z,  this->blockLength);                                            
    }      
    __aicore__ inline void Process(){    
        int16_t i = 0; 
        for ( ; i < this->tileNum; i++)Compute(i, this->tileLength);         
        Compute(this->tileNum, this->blockLength - this->tileLength * i);                                        
    }      

private:      
    __aicore__ inline void Compute(int16_t progress, uint32_t length){      
        index = progress * this->tileLength;  
        y  = Q_y.AllocTensor<half>();          
        DataCopy(y, Gm_y[index], length);     
        Q_y.EnQue(y);              
        y = Q_y.DeQue<half>();    
        Cast(tmpy, y, RoundMode::CAST_NONE, length);     
        /*核心计算->五行 常规计算->六行*/ 
        Exp(tmpy, tmpy, length);              
        // Reciprocal(tmpdy, tmpy, length);     //分子问题难解决->官方精度不够 
        Div(tmpdy, tmp025, tmpy, length);          
        AddReluCast(y, tmpdy, tmpy, length);     
        dy = Q_dy.AllocTensor<half>(); 
        DataCopy(dy, Gm_dy[index], length); 
        Q_dy.EnQue(dy);     
        dy = Q_dy.DeQue<half>();  
        Div(dy, dy, y, length); 
        Add(dy, dy, dy, length);          
        
        Q_y.FreeTensor(y);            
        Q_dy.EnQue<half>(dy);                
        dy = Q_dy.DeQue<half>();  
        DataCopy(Gm_z[index], dy, length);       
        Q_dy.FreeTensor(dy);                    
    }

private:
    TPipe* pipe;  
    TQue<QuePosition::VECIN, 2> Q_y;        
    TQue<QuePosition::VECOUT, 2> Q_dy;                              
    TBuf<QuePosition::VECCALC> tmpYBuffer, tmpDYBuffer;         
    LocalTensor<half> y, dy;         
    LocalTensor<float> tmpy, tmpdy;                        
    GlobalTensor<half> Gm_y, Gm_dy, Gm_z;        
    uint32_t index;  
    uint32_t blockLength;               
    uint16_t tileLength, tileNum;          
};

extern "C" __global__ __aicore__ void asinh_grad(GM_ADDR y, GM_ADDR dy, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) 
{       
    if(TILING_KEY_IS(1)){       
        /*Note: UB空间已塞满, 计算6行实现, 无Tiling传输, 差10us  2024/11/26   
        /*5个中间变量->搬移次数更多-> 本地：979.770813 官方：977.15 Pass*/                 
        GET_TILING_DATA(tiling_data, tiling);     
        TPipe pipe;                                                                  
        KernelAsinhGrad_Fast_Fast_V7<DTYPE_Y, DTYPE_DY, DTYPE_Z> op;                   
        op.Init(y, dy, z, tiling_data.core_size, tiling_data.block_size, tiling_data.tileNum, &pipe);                           
        op.Process();     
    }else if(TILING_KEY_IS(0)){          
        GET_TILING_DATA(tiling_data, tiling);                        
        KernelAsinhGrad_Fast<DTYPE_Y, DTYPE_DY, DTYPE_Z> op;      
        op.Init(y, dy, z, tiling_data.block_size, tiling_data.core_size);  
        op.Process();   
    }   
}   
    