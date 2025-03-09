#define K_MAX_SHAPE_DIM 0
#include "kernel_operator.h"  
using namespace AscendC;     
template<typename T> struct Map {using type = T;};      
template<> struct Map<int8_t> {using type = half;};                 
template<typename TYPE_VAR, typename TYPE_INDICES,  typename TYPE_UPDATES> class KernelScatterElements_Broadcast {
    using T = TYPE_VAR;         
public:         
    __aicore__ inline KernelScatterElements_Broadcast() {}      
    __aicore__ inline void Init(GM_ADDR var, GM_ADDR indices, GM_ADDR updates,  
                                uint32_t ALIGN_NUM, uint32_t core_size, uint32_t core_remain) {    
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!"); 
        this->blockLength = core_size + (GetBlockNum() == GetBlockIdx() + 1 ? core_remain : 0); 
        this->blockLength = this->blockLength + (this->blockLength % ALIGN_NUM ? ALIGN_NUM - this->blockLength % ALIGN_NUM : 0);  
        auto startPointer = core_size * GetBlockIdx();   
        auto bufferlength = this->blockLength;   
        Gm_var.SetGlobalBuffer((__gm__ TYPE_VAR*)var, bufferlength); 
        Gm_indices.SetGlobalBuffer((__gm__ TYPE_INDICES*)indices, bufferlength);
        Gm_updates.SetGlobalBuffer((__gm__ TYPE_UPDATES*)updates, bufferlength);      
    }   
    __aicore__ inline void Process(uint32_t shapeInf[2*5], int attrAxis, uint8_t mode) {   
        int max_dim=0;  
        if(shapeInf[0*5+0]>max_dim){   
            max_dim = shapeInf[0*5+0]; 
        }             
        if(attrAxis<0){          
            attrAxis = attrAxis+max_dim;  
        } 
        int indices_size = 1;
        for (int i = 1; i <= shapeInf[1*5+0]; ++i) { 
            indices_size *= shapeInf[1*5+i];     
        }                                            
        if(max_dim == 1){       
            for (int i = 0; i < indices_size; ++i) {
                int target_idx = (attrAxis == 0) ? Gm_indices(i) : i;
                if constexpr (std::is_same_v<T, uint8_t>) {
                    if(mode==1){       
                        Gm_var(target_idx) = static_cast<TYPE_VAR>(static_cast<int32_t>(Gm_var(target_idx)) + static_cast<int32_t>(Gm_updates(i)));
                    }else if(mode==2){    
                        Gm_var(target_idx) = static_cast<TYPE_VAR>(static_cast<int32_t>(Gm_var(target_idx)) * static_cast<int32_t>(Gm_updates(i)));
                    }else if(mode==3){    
                        Gm_var(target_idx) = Gm_updates(i);      
                    }   
                }else{     
                    if(mode==1){         
                        Gm_var(target_idx) = static_cast<TYPE_VAR>(static_cast<float>(Gm_var(target_idx)) + static_cast<float>(Gm_updates(i)));
                    }else if(mode==2){    
                        Gm_var(target_idx) = static_cast<TYPE_VAR>(static_cast<float>(Gm_var(target_idx)) * static_cast<float>(Gm_updates(i)));
                    }else if(mode==3){    
                        Gm_var(target_idx) = Gm_updates(i);      
                    }   
                }
            }
        }  
        else if (max_dim == 2) {  
            for (int i = 0; i < indices_size; ++i) {
                int target_idx[2] = {0}; 
                if (attrAxis == 0) {
                    target_idx[0] = Gm_indices(i);
                    target_idx[1] = static_cast<int>(i % shapeInf[1 * 5 + 2]);
                }else if (attrAxis == 1) {
                    target_idx[0] = static_cast<int>(i / shapeInf[1 * 5 + 2]);
                    target_idx[1] = Gm_indices(i);
                }       
                int var_offset = target_idx[0] * shapeInf[0*5+2] + target_idx[1];
                if constexpr (std::is_same_v<T, uint8_t>) {  
                    if(mode==1){        
                        Gm_var(var_offset) = static_cast<TYPE_VAR>(static_cast<int32_t>(Gm_var(var_offset)) + static_cast<int32_t>(Gm_updates(i)));
                    }else if(mode==2){    
                        Gm_var(var_offset) = static_cast<TYPE_VAR>(static_cast<int32_t>(Gm_var(var_offset)) * static_cast<int32_t>(Gm_updates(i)));
                    }else if(mode==3){    
                        Gm_var(var_offset) = Gm_updates(i);      
                    }    
                }else{
                    if(mode==1){       
                        Gm_var(var_offset) = static_cast<TYPE_VAR>(static_cast<float>(Gm_var(var_offset)) + static_cast<float>(Gm_updates(i)));
                    }else if(mode==2){    
                        Gm_var(var_offset) = static_cast<TYPE_VAR>(static_cast<float>(Gm_var(var_offset)) * static_cast<float>(Gm_updates(i)));
                    }else if(mode==3){    
                        Gm_var(var_offset) = Gm_updates(i);      
                    }   
                }   
            }
        } 
        else if (max_dim == 3) {
            for (int i = 0; i < indices_size; ++i) { 
                int target_idx[3] = {0};
                if (attrAxis == 0) {
                    target_idx[0] = Gm_indices(i);
                    target_idx[1] = static_cast<int>((i / shapeInf[1 * 5 + 3]) % shapeInf[1 * 5 + 2]);
                    target_idx[2] = static_cast<int>(i % shapeInf[1 * 5 + 3]);
                } else if (attrAxis == 1) { 
                    target_idx[0] = static_cast<int>(i / (shapeInf[1 * 5 + 2] * shapeInf[1 * 5 + 3]));
                    target_idx[1] = Gm_indices(i);
                    target_idx[2] = static_cast<int>(i % shapeInf[1 * 5 + 3]);
                } else if (attrAxis == 2) { 
                    target_idx[0] = static_cast<int>(i / (shapeInf[1 * 5 + 2] * shapeInf[1 * 5 + 3])); 
                    target_idx[1] = static_cast<int>((i / shapeInf[1 * 5 + 3]) % shapeInf[1 * 5 + 2]); 
                    target_idx[2] = Gm_indices(i); 
                }
                int var_offset = target_idx[0] * (shapeInf[0*5+2] * shapeInf[0*5+3]) + target_idx[1] * shapeInf[0*5+3] + target_idx[2];
                if constexpr (std::is_same_v<T, uint8_t>) {   
                    if(mode==1){       
                        Gm_var(var_offset) = static_cast<TYPE_VAR>(static_cast<int32_t>(Gm_var(var_offset)) + static_cast<int32_t>(Gm_updates(i)));
                    }else if(mode==2){    
                        Gm_var(var_offset) = static_cast<TYPE_VAR>(static_cast<int32_t>(Gm_var(var_offset)) * static_cast<int32_t>(Gm_updates(i)));
                    }else if(mode==3){    
                        Gm_var(var_offset) = Gm_updates(i);      
                    }    
                }else{     
                    if(mode==1){             
                        Gm_var(var_offset) = static_cast<TYPE_VAR>(static_cast<float>(Gm_var(var_offset)) + static_cast<float>(Gm_updates(i)));
                    }else if(mode==2){               
                        int32_t INT32_MAX_c = 2147483647;   // 2^31 - 1
                        int32_t INT32_MIN_c = -2147483648;  // -2^31        
                        int64_t result = static_cast<int64_t>(Gm_updates(i));           
                        if (result > INT32_MAX_c) {     
                            result = INT32_MAX_c;        
                        } else if (result < INT32_MIN_c) {
                            result = INT32_MIN_c; 
                        } 
                        Gm_var(var_offset) = static_cast<TYPE_VAR>(result);
                    }else if(mode==3){           
                        Gm_var(var_offset) = Gm_updates(i);      
                    }    
                }  
            }     
        } 
        else if (max_dim == 4) {  
            for (int i = 0; i < indices_size; ++i) {
                int target_idx[4] = {0};
                if (attrAxis == 0) {
                    target_idx[0] = Gm_indices(i);
                    target_idx[1] = static_cast<int>((i / (shapeInf[1 * 5 + 3]*shapeInf[1 * 5 + 4])) % shapeInf[1 * 5 + 2]);
                    target_idx[2] = static_cast<int>((i / shapeInf[1 * 5 + 4]) % shapeInf[1 * 5 + 3]);  
                    target_idx[3] = static_cast<int>(i % shapeInf[1 * 5 + 4]);  
                } else if (attrAxis == 1) {
                    target_idx[0] = static_cast<int>(i / (shapeInf[1 * 5 + 2] * shapeInf[1 * 5 + 3] * shapeInf[1 * 5 + 4]));
                    target_idx[1] = Gm_indices(i); 
                    target_idx[2] = static_cast<int>((i / shapeInf[1 * 5 + 4]) % shapeInf[1 * 5 + 3]); 
                    target_idx[3] = static_cast<int>(i % shapeInf[1 * 5 + 4]);    
                } else if (attrAxis == 2) {
                    target_idx[0] = static_cast<int>(i / (shapeInf[1 * 5 + 2] * shapeInf[1 * 5 + 3] * shapeInf[1 * 5 + 4]));
                    target_idx[1] = static_cast<int>((i / (shapeInf[1 * 5 + 3]*shapeInf[1 * 5 + 4])) % shapeInf[1 * 5 + 2]); 
                    target_idx[2] = Gm_indices(i); 
                    target_idx[3] = static_cast<int>(i % shapeInf[1 * 5 + 4]);    
                } else if (attrAxis == 3) {
                    target_idx[0] = static_cast<int>(i / (shapeInf[1 * 5 + 2] * shapeInf[1 * 5 + 3] * shapeInf[1 * 5 + 4]));
                    target_idx[1] = static_cast<int>((i / (shapeInf[1 * 5 + 3]*shapeInf[1 * 5 + 4])) % shapeInf[1 * 5 + 2]); 
                    target_idx[2] = static_cast<int>((i / shapeInf[1 * 5 + 4]) % shapeInf[1 * 5 + 3]); 
                    target_idx[3] = Gm_indices(i); 
                }
                int var_offset = target_idx[0] * (shapeInf[0*5+2] * shapeInf[0*5+3] * shapeInf[0*5+4]) + target_idx[1] * (shapeInf[0*5+3] * shapeInf[0*5+4]) + target_idx[2] * shapeInf[0*5+4] + target_idx[3];
                if constexpr (std::is_same_v<T, uint8_t>) {   
                    if(mode==1){       
                        Gm_var(var_offset) = static_cast<TYPE_VAR>(static_cast<int32_t>(Gm_var(var_offset)) + static_cast<int32_t>(Gm_updates(i)));
                    }else if(mode==2){    
                        Gm_var(var_offset) = static_cast<TYPE_VAR>(static_cast<int32_t>(Gm_var(var_offset)) * static_cast<int32_t>(Gm_updates(i)));
                    }else if(mode==3){      
                        Gm_var(var_offset) = Gm_updates(i);      
                    }    
                }else{  
                    if(mode==1){       
                        Gm_var(var_offset) = static_cast<TYPE_VAR>(static_cast<float>(Gm_var(var_offset)) + static_cast<float>(Gm_updates(i)));
                    }else if(mode==2){    
                        Gm_var(var_offset) = static_cast<TYPE_VAR>(static_cast<float>(Gm_var(var_offset)) * static_cast<float>(Gm_updates(i)));
                    }else if(mode==3){        
                        Gm_var(var_offset) = Gm_updates(i);      
                    }   
                }  
            }
        }
    }    
private:        
    TPipe pipe;    
    GlobalTensor<TYPE_VAR> Gm_var;     
    GlobalTensor<TYPE_INDICES> Gm_indices;  
    GlobalTensor<TYPE_UPDATES> Gm_updates;   
    uint32_t blockLength;    
};  

template<typename TYPE_VAR, typename TYPE_INDICES,  typename TYPE_UPDATES> class KernelScatterElements_FAST{
    using T = TYPE_VAR;                           
public:                                           
    __aicore__ inline KernelScatterElements_FAST() {}      
    __aicore__ inline void Init(GM_ADDR var, GM_ADDR indices, GM_ADDR updates,  
                                uint32_t core_size, uint32_t core_remain) {     
        this->blockLength = core_size + core_remain;    
        this->blockLength = this->blockLength + (this->blockLength % 8 ? 8 - this->blockLength % 8 : 0); 
        Gm_var.SetGlobalBuffer((__gm__ TYPE_VAR*)var, this->blockLength);       
        Gm_indices.SetGlobalBuffer((__gm__ TYPE_INDICES*)indices, this->blockLength);
        Gm_updates.SetGlobalBuffer((__gm__ TYPE_UPDATES*)updates, this->blockLength);      
    }               
    __aicore__ inline void Process(uint32_t *shapeInf, uint32_t F_length) {           
        for (int i = 0; i < F_length; ++i) {                                       
            Gm_var(Gm_indices(i) * (shapeInf[2] * shapeInf[3]) + 
            static_cast<int>((i / shapeInf[8]) % shapeInf[7]) * shapeInf[3] + 
            static_cast<int>(i % shapeInf[8])) = Gm_updates(i);              
        }           
    }             
private:                 
    GlobalTensor<TYPE_VAR> Gm_var;     
    GlobalTensor<TYPE_INDICES> Gm_indices;  
    GlobalTensor<TYPE_UPDATES> Gm_updates;   
    uint32_t blockLength;    
};

extern "C" __global__ __aicore__ void scatter_elements(GM_ADDR var, GM_ADDR indices,GM_ADDR updates,  GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);         
    if(TILING_KEY_IS(0)){        
        KernelScatterElements_FAST<DTYPE_VAR, DTYPE_INDICES, DTYPE_UPDATES> op;     
        op.Init(var, indices, updates, tiling_data.core_size, tiling_data.core_remain);    
        op.Process(tiling_data.shapeInf, tiling_data.F_length);   
    }else if(TILING_KEY_IS(1)){   
        KernelScatterElements_Broadcast<DTYPE_VAR, DTYPE_INDICES, DTYPE_UPDATES> op;     
        op.Init(var, indices, updates, tiling_data.ALIGN_NUM, tiling_data.core_size, tiling_data.core_remain);   
        op.Process(tiling_data.shapeInf,tiling_data.attrAxis,tiling_data.mode);  
    }
}       
     

    
 
