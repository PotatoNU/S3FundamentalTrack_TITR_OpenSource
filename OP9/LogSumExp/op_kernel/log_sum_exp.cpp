#include "kernel_operator.h"
#include <type_traits>  
using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2;
template <typename T>  
struct Map{using type = T;};
template <>struct Map<int8_t>{using type = half;};  

template <typename TYPE_X, typename TYPE_Y>
class KernelLogSumExp_Broadcast
{   
    using T = TYPE_Y; 
public:
    __aicore__ inline KernelLogSumExp_Broadcast() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y,
                                uint32_t ALIGN_NUM, uint32_t core_size, uint32_t core_remain)
    {
        this->blockLength = core_size + core_remain;                                  
        this->blockLength = this->blockLength + (this->blockLength % ALIGN_NUM ? ALIGN_NUM - this->blockLength % ALIGN_NUM : 0); 
        Gm_x.SetGlobalBuffer((__gm__ TYPE_X *)x , this->blockLength); 
        Gm_y.SetGlobalBuffer((__gm__ TYPE_Y *)y , this->blockLength);
        pipe.InitBuffer(tmp1Buffer, 8 * sizeof(float));    
        pipe.InitBuffer(tmp2Buffer, 8 * sizeof(float));   
    } 

    __aicore__ inline void Process(uint32_t shapeInf[1 * 5], int attrdim[5])
    { 
        LocalTensor<float> tmp1 = tmp1Buffer.Get<float>();  
        LocalTensor<float> tmp2 = tmp2Buffer.Get<float>();   
        uint32_t input_num = 1;
        int max_dim = 0;
        if(false){   
              
        }else{                
            Duplicate<float>(tmp2, static_cast<float>(0), 1); 
            for (int i = 0; i < input_num; i++){
                if (shapeInf[i * 5 + 0] > max_dim){
                    max_dim = shapeInf[i * 5 + 0];
                }
            }
            if (max_dim == 1){
                int max_index = 0;
                for (int i = 0; i < input_num; i++)
                {
                    if (shapeInf[i * 5 + 1] > max_index)
                    {
                        max_index = shapeInf[i * 5 + 1];
                    }
                }
                for (int i = 0; i < max_index; i++)
                { 
                    int index_x_i = (shapeInf[0 * 5 + 1] <= 1) ? 0 : i;
                    float x1_value = static_cast<float>(Gm_x(index_x_i));
                    Duplicate<float>(tmp1, static_cast<float>(x1_value), 1);
                    Exp(tmp1, tmp1, 1);   
                    Adds(tmp2, tmp2, static_cast<float>(tmp1(0)), 1); 
                } 
                Ln(tmp1, tmp2, 1); 
                Gm_y(0) = static_cast<TYPE_Y>(tmp1(0));
            }else if (max_dim == 2){      
                int max_index[2] = {}; 
                for (int i = 0; i < input_num; i++)
                {
                    for (int j = 1; j <= shapeInf[i * 5 + 0]; j++)
                    {
                        if (shapeInf[i * 5 + j] > max_index[j - 1])
                        {
                            max_index[j - 1] = shapeInf[i * 5 + j];
                        }
                    }
                } 
                if(attrdim[0]==1){
                    if (attrdim[1] == 0 || attrdim[1] == -2)
                    {     
                        for (int j = 0; j < max_index[1]; j++)
                        {
                            for (int i = 0; i < max_index[0]; i++)    
                            {
                                int index_x_i = (shapeInf[0 * 5 + 1] <= 1) ? 0 : i; 
                                int index_x_j = (shapeInf[0 * 5 + 2] <= 1) ? 0 : j; 
                                float x1_value = static_cast<float>(Gm_x(index_x_i * shapeInf[0 * 5 + 2] + index_x_j));
                                Duplicate<float>(tmp1, static_cast<float>(x1_value), 1);
                                Exp(tmp1, tmp1, 1);
                                Adds(tmp2, tmp2, static_cast<float>(tmp1(0)), 1); 
                            }
                            Ln(tmp1, tmp2, 1);
                            Gm_y(j) = static_cast<TYPE_Y>(tmp1(0));
                            Duplicate<float>(tmp2, static_cast<float>(0), 1); 
                        } 
                    }
                    else if (attrdim[1] == 1 || attrdim[1] == -1)
                    { 
                        for (int i = 0; i < max_index[0]; i++)
                        {
                            for (int j = 0; j < max_index[1]; j++)
                            {
                                int index_x_i = (shapeInf[0 * 5 + 1] <= 1) ? 0 : i;
                                int index_x_j = (shapeInf[0 * 5 + 2] <= 1) ? 0 : j; 
                                float x1_value = static_cast<float>(Gm_x(index_x_i * shapeInf[0 * 5 + 2] + index_x_j));
                                Duplicate<float>(tmp1, static_cast<float>(x1_value), 1);
                                Exp(tmp1, tmp1, 1);
                                Adds(tmp2, tmp2, static_cast<float>(tmp1(0)), 1); 
                            }
                            Ln(tmp1, tmp2, 1);
                            Gm_y(i) = static_cast<TYPE_Y>(tmp1(0));
                            Duplicate<float>(tmp2, static_cast<float>(0), 1); 
                        }
                    } 
                }else{ 
                    for (int j = 0; j < max_index[1]; j++)
                    { 
                        for (int i = 0; i < max_index[0]; i++)
                        {
                            int index_x_i = (shapeInf[0 * 5 + 1] <= 1) ? 0 : i; 
                            int index_x_j = (shapeInf[0 * 5 + 2] <= 1) ? 0 : j; 
                            float x1_value = static_cast<float>(Gm_x(index_x_i * shapeInf[0 * 5 + 2] + index_x_j));
                            Duplicate<float>(tmp1, static_cast<float>(x1_value), 1);
                            Exp(tmp1, tmp1, 1);
                            Adds(tmp2, tmp2, static_cast<float>(tmp1(0)), 1); 
                        }
                    } 
                    Ln(tmp1, tmp2, 1);
                    Gm_y(0) = static_cast<TYPE_Y>(tmp1(0));    
                }       
            }         
            else if (max_dim == 3){    
                int max_index[3] = {}; 
                for (int i = 0; i < input_num; i++){
                    for (int j = 1; j <= shapeInf[i * 5 + 0]; j++){
                        if (shapeInf[i * 5 + j] > max_index[j - 1]){
                            max_index[j - 1] = shapeInf[i * 5 + j];
                        }    
                    }        
                } 
                if(attrdim[0]==1){     
                    if (attrdim[1] == 0 || attrdim[1] == -3)
                    {    
                        float x1_value = 0; 
                        for (int j = 0; j < max_index[1]; j++) {        
                            for (int k = 0; k < max_index[2]; k++){ 
                                for (int i = 0; i < max_index[0]; i++){         
                                    x1_value = static_cast<float>(Gm_x(i * shapeInf[0 * 5 + 2] * shapeInf[0 * 5 + 3] + j * shapeInf[0 * 5 + 3] + k));  
                                    tmp1(0) = x1_value;  
                                    Exp(tmp1, tmp1, 1);               
                                    Adds(tmp2, tmp2, static_cast<float>(tmp1(0)), 1);   
                                }           
                                Ln(tmp1, tmp2, 1);               
                                Gm_y(j * max_index[2] + k) = static_cast<TYPE_Y>(tmp1(0));          
                                tmp2(0) = static_cast<float>(0);             
                            }
                        }    
                    }        
                    else if (attrdim[1] == 1 || attrdim[1] == -2)
                    {
                        for (int i = 0; i < max_index[0]; i++)
                        {
                            for (int k = 0; k < max_index[2]; k++)
                            {
                                for (int j = 0; j < max_index[1]; j++)
                                {
                                    int index_x_i = (shapeInf[0 * 5 + 1] <= 1) ? 0 : i;
                                    int index_x_j = (shapeInf[0 * 5 + 2] <= 1) ? 0 : j;
                                    int index_x_k = (shapeInf[0 * 5 + 3] <= 1) ? 0 : k;

                                    float x1_value = static_cast<float>(Gm_x(index_x_i * shapeInf[0 * 5 + 2] * shapeInf[0 * 5 + 3] + index_x_j * shapeInf[0 * 5 + 3] + index_x_k));
                                    Duplicate<float>(tmp1, static_cast<float>(x1_value), 1);
                                    Exp(tmp1, tmp1, 1);
                                    Adds(tmp2, tmp2, static_cast<float>(tmp1(0)), 1); 
                                }
                                Ln(tmp1, tmp2, 1);
                                Gm_y(i * max_index[2] + k) = static_cast<TYPE_Y>(tmp1(0));
                                Duplicate<float>(tmp2, static_cast<float>(0), 1); 
                            }
                        }
                    }
                    else if (attrdim[1] == 2 || attrdim[1] == -1)
                    { 
                        for (int i = 0; i < max_index[0]; i++)
                        {
                            for (int j = 0; j < max_index[1]; j++) 
                            {
                                for (int k = 0; k < max_index[2]; k++)
                                {
                                    int index_x_i = (shapeInf[0 * 5 + 1] <= 1) ? 0 : i;
                                    int index_x_j = (shapeInf[0 * 5 + 2] <= 1) ? 0 : j;
                                    int index_x_k = (shapeInf[0 * 5 + 3] <= 1) ? 0 : k;
                                    float x1_value = static_cast<float>(Gm_x(index_x_i * shapeInf[0 * 5 + 2] * shapeInf[0 * 5 + 3] + index_x_j * shapeInf[0 * 5 + 3] + index_x_k));
                                    Duplicate<float>(tmp1, static_cast<float>(x1_value), 1);
                                    Exp(tmp1, tmp1, 1);
                                    Adds(tmp2, tmp2, static_cast<float>(tmp1(0)), 1); 
                                }
                                Ln(tmp1, tmp2, 1);
                                Gm_y(i * max_index[1] + j) = static_cast<TYPE_Y>(tmp1(0));
                                Duplicate<float>(tmp2, static_cast<float>(0), 1);
                            }
                        }
                    }
                }  
                else if(attrdim[0]==2){                      
                    if(((attrdim[1]==0 || attrdim[1]==-3) && (attrdim[2]==1  || attrdim[2]==-2)) ||
                       ((attrdim[1]==1 || attrdim[1]==-2) && (attrdim[2]==0) || attrdim[2]==-3)){    
                        for (int k = 0; k < max_index[2]; k++) {           
                            Duplicate<float>(tmp2, static_cast<float>(0), 1);
                            for (int i = 0; i < max_index[0]; i++) {
                                for (int j = 0; j < max_index[1]; j++) {
                                    int index_x_i = (shapeInf[0 * 5 + 1] <= 1) ? 0 : i;
                                    int index_x_j = (shapeInf[0 * 5 + 2] <= 1) ? 0 : j;
                                    int index_x_k = (shapeInf[0 * 5 + 3] <= 1) ? 0 : k;
                                    float x1_value = static_cast<float>(Gm_x(index_x_i * shapeInf[0 * 5 + 2] * shapeInf[0 * 5 + 3] + index_x_j * shapeInf[0 * 5 + 3] + index_x_k));
                                    Duplicate<float>(tmp1, static_cast<float>(x1_value), 1); 
                                    Exp(tmp1, tmp1, 1);
                                    Adds(tmp2, tmp2, static_cast<float>(tmp1(0)), 1); 
                                }
                            }
                            Ln(tmp1, tmp2, 1);
                            Gm_y(k) = static_cast<TYPE_Y>(tmp1(0));
                        } 
                    }     
                    else if(((attrdim[1]==0 || attrdim[1]==-3) && (attrdim[2]==2 || attrdim[2]==-1)) ||
                            ((attrdim[1]==2 || attrdim[1]==-1) && (attrdim[2]==0) || attrdim[2]==-3)){   
                        for (int j = 0; j < max_index[1]; j++) {
                            Duplicate<float>(tmp2, static_cast<float>(0), 1); 
                            for (int i = 0; i < max_index[0]; i++) {
                                for (int k = 0; k < max_index[2]; k++) {
                                    int index_x_i = (shapeInf[0 * 5 + 1] <= 1) ? 0 : i;
                                    int index_x_j = (shapeInf[0 * 5 + 2] <= 1) ? 0 : j;
                                    int index_x_k = (shapeInf[0 * 5 + 3] <= 1) ? 0 : k;
                                    float x1_value = static_cast<float>(Gm_x(index_x_i * shapeInf[0 * 5 + 2] * shapeInf[0 * 5 + 3] + index_x_j * shapeInf[0 * 5 + 3] + index_x_k));
                                    Duplicate<float>(tmp1, static_cast<float>(x1_value), 1); 
                                    Exp(tmp1, tmp1, 1);
                                    Adds(tmp2, tmp2, static_cast<float>(tmp1(0)), 1); 
                                }
                            }
                            Ln(tmp1, tmp2, 1);
                            Gm_y(j) = static_cast<TYPE_Y>(tmp1(0));
                        } 
                    }
                    else if(((attrdim[1]==1 || attrdim[1]==-2) && (attrdim[2]==2 || attrdim[2]==-1)) ||
                            ((attrdim[1]==2 || attrdim[1]==-1) && (attrdim[2]==1) || attrdim[2]==-2)){        
                        for (int i = 0; i < max_index[0]; i++) { 
                            Duplicate<float>(tmp2, static_cast<float>(0), 1); 
                            for (int j = 0; j < max_index[1]; j++) {
                                for (int k = 0; k < max_index[2]; k++) {
                                    int index_x_i = (shapeInf[0 * 5 + 1] <= 1) ? 0 : i; 
                                    int index_x_j = (shapeInf[0 * 5 + 2] <= 1) ? 0 : j;
                                    int index_x_k = (shapeInf[0 * 5 + 3] <= 1) ? 0 : k;
                                    float x1_value = static_cast<float>(Gm_x(index_x_i * shapeInf[0 * 5 + 2] * shapeInf[0 * 5 + 3] + index_x_j * shapeInf[0 * 5 + 3] + index_x_k));
                                    Duplicate<float>(tmp1, static_cast<float>(x1_value), 1); 
                                    Exp(tmp1, tmp1, 1);
                                    Adds(tmp2, tmp2, static_cast<float>(tmp1(0)), 1); 
                                }
                            }
                            Ln(tmp1, tmp2, 1);
                            Gm_y(i) = static_cast<TYPE_Y>(tmp1(0));
                        }
                    }  
                }              
            }   
        }            
    }     

private:
    TPipe pipe;
    GlobalTensor<TYPE_X> Gm_x;
    GlobalTensor<TYPE_Y> Gm_y;
    TBuf<QuePosition::VECCALC> tmp1Buffer, tmp2Buffer; 
    uint32_t blockLength; 
};

/*处理 [c,h,w] dim=0 的情况，其中基底h*w*size(TYPE_X) 可大于ub_size, 需对底部进一步切片*/ 
template <typename TYPE_X, typename TYPE_Y>
class KernelLogSumExp_Broadcast_TwoDim_V2   
{    
    using T = TYPE_Y;  
public: 
    __aicore__ inline KernelLogSumExp_Broadcast_TwoDim_V2() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y,
                                uint32_t core_size, uint32_t core_remain, TPipe* pipeIn,
                                uint32_t shapeInf[1 * 5])
    {
        pipe = pipeIn;    
        this->blockLength = core_size + core_remain;                                                    
        this->blockLength = this->blockLength + (this->blockLength % 8 ? 8 - this->blockLength % 8 : 0);
        Gm_x.SetGlobalBuffer((__gm__ TYPE_X *)x , this->blockLength);  
        Gm_y.SetGlobalBuffer((__gm__ TYPE_Y *)y , this->blockLength);  
        this->botileNum  = 256;                                 
        this->tileLength = shapeInf[2]*shapeInf[3] / this->botileNum;               
        this->tileLength = this->tileLength + (this->tileLength % 8 ? 8 : 0);    
        this->tileNum = shapeInf[1] * this->botileNum;            

        pipe->InitBuffer(Q_x, BUFFER_NUM, this->tileLength * 4);  
        pipe->InitBuffer(Q_y, BUFFER_NUM, this->tileLength * 4);   
        pipe->InitBuffer(tmp1Buffer, this->tileLength * 4);                    
    } 

    __aicore__ inline void Process(uint32_t shapeInf[1 * 5]) 
    { 
        LocalTensor<TYPE_X> tmp1 = tmp1Buffer.Get<TYPE_X>();   
        // 底层切片分组概念->处理大块底层数据 
        for (int32_t bo = 0; bo < this->botileNum; bo++) {           
            for (int32_t i = 0+bo; i < this->tileNum; i=i+this->botileNum) {   
                LocalTensor<TYPE_X>  x = Q_x.AllocTensor<TYPE_X>();    
                DataCopy(x, Gm_x[i * (this->tileLength - (this->tileLength % 8 ? 8 : 0))], this->tileLength);  
                Q_x.EnQue(x);         
                x = Q_x.DeQue<TYPE_X>();   
                if(i == 0+bo){         
                    Exp(tmp1, x, this->tileLength);                 
                }else{          
                    Exp(x, x, this->tileLength);      
                    Add(tmp1, x, tmp1, this->tileLength);      
                    if(i == this->tileNum+(bo-this->botileNum)){                        
                        LocalTensor<TYPE_Y> y = Q_y.AllocTensor<TYPE_Y>();      
                        Ln(y, tmp1, this->tileLength);      
                        Q_y.EnQue<TYPE_Y>(y);      
                        y = Q_y.DeQue<TYPE_Y>();        
                        DataCopy(Gm_y[bo * (this->tileLength - (this->tileLength % 8 ? 8 : 0))], y, this->tileLength);  
                        Q_y.FreeTensor(y);
                    }   
                }    
                Q_x.FreeTensor(x);  
            }
        }       
    }     

private:
    TPipe* pipe;
    GlobalTensor<TYPE_X> Gm_x,Gm_y;  
    TQue<QuePosition::VECIN, BUFFER_NUM> Q_x;    
    TQue<QuePosition::VECOUT, BUFFER_NUM> Q_y;
    TBuf<QuePosition::VECCALC> tmp1Buffer; 
    uint32_t tileLength;  
    uint32_t tileNum; 
    uint32_t botileNum;   //底层切片份数 
    uint32_t blockLength; 
};

// 单平面搬入，一次rep处理4行  -> 2*4=8 DateBlock
template <typename TYPE_X, typename TYPE_Y>
class KernelLogSumExp_Broadcast_OneDim_V8   
{     
    using T = TYPE_Y; 
public:
    __aicore__ inline KernelLogSumExp_Broadcast_OneDim_V8() {} 
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y,
                                uint32_t ALIGN_NUM, uint32_t core_size, uint32_t core_remain, TPipe* pipeIn, 
                                uint32_t shapeInf[1 * 5])    
    {
        pipe = pipeIn;     
        this->blockLength = core_size + core_remain;                                          
        this->blockLength = this->blockLength + (this->blockLength % ALIGN_NUM ? ALIGN_NUM - this->blockLength % ALIGN_NUM : 0); 
        this->tileLength = shapeInf[3]*shapeInf[4];                  
        this->tileLength = this->tileLength + (this->tileLength % ALIGN_NUM ? ALIGN_NUM - this->tileLength % ALIGN_NUM : 0);  
        Gm_x.SetGlobalBuffer((__gm__ TYPE_X *)x , this->blockLength);       
        Gm_y.SetGlobalBuffer((__gm__ TYPE_Y *)y , this->blockLength);                                                              
        pipe->InitBuffer(Q_x, BUFFER_NUM, this->tileLength * sizeof(TYPE_X));     
        pipe->InitBuffer(Q_y, BUFFER_NUM,      shapeInf[4] * sizeof(TYPE_Y));          
        pipe->InitBuffer(tmp1Buffer,         4*shapeInf[4] * sizeof(TYPE_X));     //存每片w轴累加量->整体累加量                
        tmp1 = tmp1Buffer.Get<TYPE_X>();    
    } 

    __aicore__ inline void Process(uint32_t shapeInf[1 * 5])
    {                       
        /*w轴取首地址0，无需遍历*/      
        for (int c = 0; c < shapeInf[2]; c++) {        /*C-1*/    
            for (int n = 0; n < shapeInf[1]; n++) {    /*N-0*/   
                // n*h求和项清零 
                if(n==0)Duplicate<TYPE_X>(tmp1, TYPE_X(0), 4*shapeInf[4]);            
                int index = n * shapeInf[0 * 5 + 2] * shapeInf[0 * 5 + 3] * shapeInf[0 * 5 + 4] +
                            c * shapeInf[0 * 5 + 3] * shapeInf[0 * 5 + 4]; 
                // 0、大平面搬入->平面求EXP
                LocalTensor<TYPE_X>  x = Q_x.AllocTensor<TYPE_X>();       
                DataCopy(x, Gm_x[index], this->tileLength);  
                Q_x.EnQue(x);          
                x = Q_x.DeQue<TYPE_X>();        
                Exp(x, x, this->tileLength);       
                
                // 1、整块搬入，一次调用求和(512行,一次rep处理2行, 512/2-1=255次rep，为一条API极限)
                uint64_t mask = 4*shapeInf[4];    //单次迭代处理数据量-> fp32数据,64以内->16*2=32个 两行每次    
                Add(x,x,x[4*shapeInf[4]],      
                    mask, 512/4-1, { 1, 1, 1, 0, 0, uint8_t(4*shapeInf[4]/8)});             
                Add(tmp1, x, tmp1, 4*shapeInf[4]);          
                Q_x.FreeTensor(x);      

                //2、最后处理    
                if(n==(shapeInf[1]-1)){      
                    Add(tmp1[0*shapeInf[4]+0],  
                        tmp1[0*shapeInf[4]+0], 
                        tmp1[2*shapeInf[4]+0], 2*shapeInf[4]);    

                    Add(tmp1[0*shapeInf[4]+0],  
                        tmp1[0*shapeInf[4]+0],  
                        tmp1[1*shapeInf[4]+0], 1*shapeInf[4]);                           

                    LocalTensor<TYPE_Y> y = Q_y.AllocTensor<TYPE_Y>();      
                    Ln(y, tmp1, shapeInf[4]);             
                    Q_y.EnQue<TYPE_Y>(y);      
                    y = Q_y.DeQue<TYPE_Y>();            
                    DataCopy(Gm_y[c * shapeInf[4] + 0], y, shapeInf[4]);    
                    Q_y.FreeTensor(y);     
                }        
            }/*N H 两通道内值完成求和*/ 
        }           
    }     
    
private:
    TPipe* pipe; 
    GlobalTensor<TYPE_X> Gm_x,Gm_y; 
    TQue<QuePosition::VECIN, BUFFER_NUM> Q_x;       
    TQue<QuePosition::VECOUT, BUFFER_NUM> Q_y;   
    TBuf<QuePosition::VECCALC> tmp1Buffer; 
    LocalTensor<TYPE_X> tmp1;
    uint32_t blockLength, tileLength;         
};
 
extern "C" __global__ __aicore__ void log_sum_exp(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling){   
     
    if(TILING_KEY_IS(5)){        
        GET_TILING_DATA(tiling_data, tiling);      
        TPipe pipe;                                
        KernelLogSumExp_Broadcast_OneDim_V8<DTYPE_X, DTYPE_Y> op;      
        op.Init(x, y,          
                tiling_data.ALIGN_NUM, tiling_data.core_size, tiling_data.core_remain,
                &pipe, tiling_data.shapeInf);                    
        op.Process(tiling_data.shapeInf);           
        
    }else if(TILING_KEY_IS(3)){     
        GET_TILING_DATA(tiling_data, tiling);   
        TPipe pipe;       
        KernelLogSumExp_Broadcast_TwoDim_V2<DTYPE_X, DTYPE_Y> op;     
        op.Init(x, y,       
                tiling_data.core_size, tiling_data.core_remain,&pipe,
                tiling_data.shapeInf);     
        op.Process(tiling_data.shapeInf);               
    }else if(TILING_KEY_IS(1)){      
        GET_TILING_DATA(tiling_data, tiling);   
        KernelLogSumExp_Broadcast<DTYPE_X, DTYPE_Y> op;  
        op.Init(x, y,                                                    
                tiling_data.ALIGN_NUM, tiling_data.core_size, tiling_data.core_remain);
        op.Process(tiling_data.shapeInf, tiling_data.attrdim);
    }    
}  
