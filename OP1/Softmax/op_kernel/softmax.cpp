#include "kernel_operator.h"
#include <type_traits>                           
#define LightNum 32                      
#define BlockLength (16*128*480)                  
#define TileLength (LightNum*480)           
#define TileNum (BlockLength/TileLength) 
#define LightRemainNum ((BlockLength-TileNum*TileLength)/480)  
#define TileRemain (LightRemainNum*480)                             
using namespace AscendC;                   
constexpr int32_t BUFFER_NUM = 2;                 
template<typename T> struct Map {using type = T;};      
template<> struct Map<int8_t> {using type = half;};                   
template<typename TYPE_X, typename TYPE_Y> class KernelSoftmax_Norm { 
    using T = TYPE_Y;         
public:
    __aicore__ inline KernelSoftmax_Norm() {}       
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y,    
                                uint32_t core_size) {    
        this->blockLength = core_size; 
        Gm_x.SetGlobalBuffer((__gm__ TYPE_X*)x, this->blockLength); 
        Gm_y.SetGlobalBuffer((__gm__ TYPE_Y*)y, this->blockLength);     
        pipe.InitBuffer(tmp1Buffer, 1 * sizeof(DTYPE_Y));   
        pipe.InitBuffer(tmp2Buffer, 1 * sizeof(DTYPE_Y));   
        pipe.InitBuffer(tmp32_1Buffer, 1 * sizeof(float));    
        pipe.InitBuffer(tmp32_2Buffer, 1 * sizeof(float));     
    }    
    __aicore__ inline void Process(uint32_t shapeInf[1*5], int attrdim) {      
        LocalTensor<TYPE_Y> tmp1 = tmp1Buffer.Get<TYPE_Y>();     
        LocalTensor<TYPE_Y> tmp2 = tmp2Buffer.Get<TYPE_Y>(); 
        LocalTensor<float> tmp32_1 = tmp32_1Buffer.Get<float>();     
        LocalTensor<float> tmp32_2 = tmp32_2Buffer.Get<float>();   
        uint32_t input_num=1;           
        int max_dim=0; 
        Duplicate<TYPE_Y>(tmp2, static_cast<TYPE_Y>(0), 1);     
        for(int i=0;i<input_num;i++){ 
            if(shapeInf[i*5+0]>max_dim){  
                max_dim = shapeInf[i*5+0];  
            }    
        }    
        if (max_dim == 1) {                   
            int max_index = 0;  
            for (int i = 0; i < input_num; i++) {
                if (shapeInf[i * 5 + 1] > max_index) { 
                    max_index = shapeInf[i * 5 + 1]; 
                }      
            }  
            for (int i = 0; i < max_index; i++) {     
                int index_x_i = (shapeInf[0 * 5 + 1] <= 1) ? 0 : i;   
                TYPE_Y x1_value  = static_cast<float>(Gm_x(index_x_i));  
                Duplicate<TYPE_Y>(tmp1, static_cast<TYPE_Y>(x1_value), 1);    
                Exp(tmp1, tmp1, 1);
                Adds(tmp2, tmp2, static_cast<TYPE_Y>(tmp1(0)), 1);  
            }     
            for (int i = 0; i < max_index; i++) {      
                int index_x_i = (shapeInf[0 * 5 + 1] <= 1) ? 0 : i;      
                TYPE_Y x1_value  = static_cast<float>(Gm_x(index_x_i));    
                Duplicate<TYPE_Y>(tmp1, static_cast<TYPE_Y>(x1_value), 1);    
                Exp(tmp1, tmp1, 1);  
                Div(tmp1, tmp1, tmp2, 1);       
                Gm_y(i) = static_cast<TYPE_Y>(tmp1(0));       
            }     
        }   
        else if (max_dim == 2) {   
            int max_index[2] = {};  
            for (int i = 0; i < input_num; i++) { 
                for (int j = 1; j <= shapeInf[i * 5 + 0]; j++) {  
                    if (shapeInf[i * 5 + j] > max_index[j - 1]) {
                        max_index[j - 1] = shapeInf[i * 5 + j];
                    }
                }  
            }    
            if(attrdim == 0){  
                for (int j = 0; j < max_index[1]; j++){           
                    for (int i = 0; i < max_index[0]; i++){   
                        int index_x_i = (shapeInf[0 * 5 + 1] <= 1) ? 0 : i;  
                        int index_x_j = (shapeInf[0 * 5 + 2] <= 1) ? 0 : j; 
                        TYPE_Y x1_value  = static_cast<float>(Gm_x(index_x_i  * shapeInf[0 * 5 + 2] + index_x_j));   
                        Duplicate<TYPE_Y>(tmp1, static_cast<TYPE_Y>(x1_value), 1);  
                        Exp(tmp1, tmp1, 1);                                                            
                        Adds(tmp2, tmp2, static_cast<TYPE_Y>(tmp1(0)), 1);  
                    }
                    for (int i = 0; i < max_index[0]; i++){    
                        int index_x_i = (shapeInf[0 * 5 + 1] <= 1) ? 0 : i; 
                        int index_x_j = (shapeInf[0 * 5 + 2] <= 1) ? 0 : j;  
                        TYPE_Y x1_value  = static_cast<float>(Gm_x(index_x_i  * shapeInf[0 * 5 + 2] + index_x_j));   
                        Duplicate<TYPE_Y>(tmp1, static_cast<TYPE_Y>(x1_value), 1);    
                        Exp(tmp1, tmp1, 1);  
                        Div(tmp1, tmp1, tmp2, 1);       
                        Gm_y(i * max_index[1] + j) = static_cast<TYPE_Y>(tmp1(0));       
                    }
                    Duplicate<TYPE_Y>(tmp2, static_cast<TYPE_Y>(0), 1);    
                }      
            } 
            else if(attrdim == 1 || attrdim == -1){
                for (int i = 0; i < max_index[0]; i++){           
                    for (int j = 0; j < max_index[1]; j++){   
                        int index_x_i = (shapeInf[0 * 5 + 1] <= 1) ? 0 : i;  
                        int index_x_j = (shapeInf[0 * 5 + 2] <= 1) ? 0 : j;  
                        TYPE_Y x1_value  = static_cast<float>(Gm_x(index_x_i  * shapeInf[0 * 5 + 2] + index_x_j));   
                        Duplicate<TYPE_Y>(tmp1, static_cast<TYPE_Y>(x1_value), 1);    
                        Exp(tmp1, tmp1, 1);
                        Adds(tmp2, tmp2, static_cast<TYPE_Y>(tmp1(0)), 1); 
                    }
                    for (int j = 0; j < max_index[1]; j++){    
                        int index_x_i = (shapeInf[0 * 5 + 1] <= 1) ? 0 : i;  
                        int index_x_j = (shapeInf[0 * 5 + 2] <= 1) ? 0 : j; 
                        TYPE_Y x1_value  = static_cast<float>(Gm_x(index_x_i  * shapeInf[0 * 5 + 2] + index_x_j));   
                        Duplicate<TYPE_Y>(tmp1, static_cast<TYPE_Y>(x1_value), 1);    
                        Exp(tmp1, tmp1, 1);  
                        Div(tmp1, tmp1, tmp2, 1);       
                        Gm_y(i * max_index[1] + j) = static_cast<TYPE_Y>(tmp1(0));       
                    }
                    Duplicate<TYPE_Y>(tmp2, static_cast<TYPE_Y>(0), 1);    
                }      
            } 
        }     
        else if (max_dim == 3) {          
            int max_index[3] = {};
            for (int i = 0; i < input_num; i++) {
                for (int j = 1; j <= shapeInf[i * 5 + 0]; j++) {
                    if (shapeInf[i * 5 + j] > max_index[j - 1]) {
                        max_index[j - 1] = shapeInf[i * 5 + j];
                    }
                }
            }
            if (attrdim == 0) {
                for (int j = 0; j < max_index[1]; j++) {
                    for (int k = 0; k < max_index[2]; k++) {
                        for (int i = 0; i < max_index[0]; i++) {
                            int index_x_i = (shapeInf[0 * 5 + 1] <= 1) ? 0 : i;
                            int index_x_j = (shapeInf[0 * 5 + 2] <= 1) ? 0 : j;
                            int index_x_k = (shapeInf[0 * 5 + 3] <= 1) ? 0 : k;

                            TYPE_Y x1_value = static_cast<float>(Gm_x(index_x_i * shapeInf[0 * 5 + 2] * shapeInf[0 * 5 + 3] + index_x_j * shapeInf[0 * 5 + 3] + index_x_k));
                            Duplicate<TYPE_Y>(tmp1, static_cast<TYPE_Y>(x1_value), 1);
                            Exp(tmp1, tmp1, 1);
                            Adds(tmp2, tmp2, static_cast<TYPE_Y>(tmp1(0)), 1);  
                        }
                        for (int i = 0; i < max_index[0]; i++) {
                            int index_x_i = (shapeInf[0 * 5 + 1] <= 1) ? 0 : i;
                            int index_x_j = (shapeInf[0 * 5 + 2] <= 1) ? 0 : j;
                            int index_x_k = (shapeInf[0 * 5 + 3] <= 1) ? 0 : k;

                            TYPE_Y x1_value = static_cast<float>(Gm_x(index_x_i * shapeInf[0 * 5 + 2] * shapeInf[0 * 5 + 3] + index_x_j * shapeInf[0 * 5 + 3] + index_x_k));
                            Duplicate<TYPE_Y>(tmp1, static_cast<TYPE_Y>(x1_value), 1);
                            Exp(tmp1, tmp1, 1);
                            Div(tmp1, tmp1, tmp2, 1);
                            Gm_y(i * max_index[1] * max_index[2] + j * max_index[2] + k) = static_cast<TYPE_Y>(tmp1(0));
                        }
                        Duplicate<TYPE_Y>(tmp2, static_cast<TYPE_Y>(0), 1); 
                    }
                }
            }
            else if (attrdim == 1 || attrdim == -2) {
                for (int i = 0; i < max_index[0]; i++) {
                    for (int k = 0; k < max_index[2]; k++) {
                        for (int j = 0; j < max_index[1]; j++) {
                            int index_x_i = (shapeInf[0 * 5 + 1] <= 1) ? 0 : i;
                            int index_x_j = (shapeInf[0 * 5 + 2] <= 1) ? 0 : j;
                            int index_x_k = (shapeInf[0 * 5 + 3] <= 1) ? 0 : k;

                            TYPE_Y x1_value = static_cast<float>(Gm_x(index_x_i * shapeInf[0 * 5 + 2] * shapeInf[0 * 5 + 3] + index_x_j * shapeInf[0 * 5 + 3] + index_x_k));
                            Duplicate<TYPE_Y>(tmp1, static_cast<TYPE_Y>(x1_value), 1);
                            Exp(tmp1, tmp1, 1);
                            Adds(tmp2, tmp2, static_cast<TYPE_Y>(tmp1(0)), 1); 
                        }
                        for (int j = 0; j < max_index[1]; j++) {
                            int index_x_i = (shapeInf[0 * 5 + 1] <= 1) ? 0 : i;
                            int index_x_j = (shapeInf[0 * 5 + 2] <= 1) ? 0 : j;
                            int index_x_k = (shapeInf[0 * 5 + 3] <= 1) ? 0 : k;

                            TYPE_Y x1_value = static_cast<float>(Gm_x(index_x_i * shapeInf[0 * 5 + 2] * shapeInf[0 * 5 + 3] + index_x_j * shapeInf[0 * 5 + 3] + index_x_k));
                            Duplicate<TYPE_Y>(tmp1, static_cast<TYPE_Y>(x1_value), 1);
                            Exp(tmp1, tmp1, 1);
                            Div(tmp1, tmp1, tmp2, 1);
                            Gm_y(i * max_index[1] * max_index[2] + j * max_index[2] + k) = static_cast<TYPE_Y>(tmp1(0));
                        }
                        Duplicate<TYPE_Y>(tmp2, static_cast<TYPE_Y>(0), 1);  
                    }
                }
            }
            else if (attrdim == 2 || attrdim == -1) {      
                for (int i = 0; i < max_index[0]; i++) { 
                    for (int j = 0; j < max_index[1]; j++) {
                        if constexpr (std::is_same_v<T, half>){ 
                            for (int k = 0; k < max_index[2]; k++) { 
                                int index_x_i = (shapeInf[0 * 5 + 1] <= 1) ? 0 : i;
                                int index_x_j = (shapeInf[0 * 5 + 2] <= 1) ? 0 : j;
                                int index_x_k = (shapeInf[0 * 5 + 3] <= 1) ? 0 : k;

                                float x1_value = static_cast<float>(Gm_x(index_x_i * shapeInf[0 * 5 + 2] * shapeInf[0 * 5 + 3] + index_x_j * shapeInf[0 * 5 + 3] + index_x_k));
                                Duplicate<float>(tmp32_1, static_cast<float>(x1_value), 1);
                                Exp(tmp32_1, tmp32_1, 1);
                                Adds(tmp32_2, tmp32_2, static_cast<float>(tmp32_1(0)), 1);  
                            }
                            for (int k = 0; k < max_index[2]; k++) { 
                                int index_x_i = (shapeInf[0 * 5 + 1] <= 1) ? 0 : i;
                                int index_x_j = (shapeInf[0 * 5 + 2] <= 1) ? 0 : j;
                                int index_x_k = (shapeInf[0 * 5 + 3] <= 1) ? 0 : k;
                                float x1_value = static_cast<float>(Gm_x(index_x_i * shapeInf[0 * 5 + 2] * shapeInf[0 * 5 + 3] + index_x_j * shapeInf[0 * 5 + 3] + index_x_k));
                                Duplicate<float>(tmp32_1, static_cast<float>(x1_value), 1);
                                Exp(tmp32_1, tmp32_1, 1); 
                                Div(tmp32_1, tmp32_1, tmp32_2, 1);  
                                Cast(tmp1, tmp32_1, RoundMode::CAST_ROUND, 1);   
                                Gm_y(i * max_index[1] * max_index[2] + j * max_index[2] + k) = static_cast<TYPE_Y>(tmp1(0)); 
                            }
                            Duplicate<float>(tmp32_2, static_cast<float>(0), 1);  
                        }else{
                            for (int k = 0; k < max_index[2]; k++) { 
                                int index_x_i = (shapeInf[0 * 5 + 1] <= 1) ? 0 : i;
                                int index_x_j = (shapeInf[0 * 5 + 2] <= 1) ? 0 : j;
                                int index_x_k = (shapeInf[0 * 5 + 3] <= 1) ? 0 : k;

                                TYPE_Y x1_value = static_cast<TYPE_Y>(Gm_x(index_x_i * shapeInf[0 * 5 + 2] * shapeInf[0 * 5 + 3] + index_x_j * shapeInf[0 * 5 + 3] + index_x_k));
                                Duplicate<TYPE_Y>(tmp1, static_cast<TYPE_Y>(x1_value), 1);
                                Exp(tmp1, tmp1, 1);
                                Adds(tmp2, tmp2, static_cast<TYPE_Y>(tmp1(0)), 1);  
                            }
                            for (int k = 0; k < max_index[2]; k++) {
                                int index_x_i = (shapeInf[0 * 5 + 1] <= 1) ? 0 : i;
                                int index_x_j = (shapeInf[0 * 5 + 2] <= 1) ? 0 : j;
                                int index_x_k = (shapeInf[0 * 5 + 3] <= 1) ? 0 : k;

                                TYPE_Y x1_value = static_cast<TYPE_Y>(Gm_x(index_x_i * shapeInf[0 * 5 + 2] * shapeInf[0 * 5 + 3] + index_x_j * shapeInf[0 * 5 + 3] + index_x_k));
                                Duplicate<TYPE_Y>(tmp1, static_cast<TYPE_Y>(x1_value), 1);
                                Exp(tmp1, tmp1, 1);
                                Div(tmp1, tmp1, tmp2, 1);  
                                Gm_y(i * max_index[1] * max_index[2] + j * max_index[2] + k) = static_cast<TYPE_Y>(tmp1(0));
                            }
                            Duplicate<TYPE_Y>(tmp2, static_cast<TYPE_Y>(0), 1); 
                        }
                    }  
                }
            }
        }  
        else if (max_dim == 4) { 
            int max_index[4] = {};
            for (int i = 0; i < input_num; i++) {
                for (int j = 1; j <= shapeInf[i * 5 + 0]; j++) {
                    if (shapeInf[i * 5 + j] > max_index[j - 1]) {
                        max_index[j - 1] = shapeInf[i * 5 + j];
                    }
                }
            }
            if (attrdim == 0) { 
                for (int j = 0; j < max_index[1]; j++) {
                    for (int k = 0; k < max_index[2]; k++) {
                        for (int l = 0; l < max_index[3]; l++) {
                            float sum = 0;
                            for (int i = 0; i < max_index[0]; i++) {
                                int index_x_i = (shapeInf[0 * 5 + 1] <= 1) ? 0 : i;
                                int index_x_j = (shapeInf[0 * 5 + 2] <= 1) ? 0 : j;
                                int index_x_k = (shapeInf[0 * 5 + 3] <= 1) ? 0 : k;
                                int index_x_l = (shapeInf[0 * 5 + 4] <= 1) ? 0 : l;

                                TYPE_Y x1_value = static_cast<TYPE_Y>(Gm_x(index_x_i * shapeInf[0 * 5 + 2] * shapeInf[0 * 5 + 3] * shapeInf[0 * 5 + 4] + index_x_j * shapeInf[0 * 5 + 3] * shapeInf[0 * 5 + 4] + index_x_k * shapeInf[0 * 5 + 4] + index_x_l));
                                Duplicate<TYPE_Y>(tmp1, static_cast<TYPE_Y>(x1_value), 1);
                                Exp(tmp1, tmp1, 1);
                                Adds(tmp2, tmp2, static_cast<TYPE_Y>(tmp1(0)), 1); 
                            }
                            for (int i = 0; i < max_index[0]; i++) {
                                int index_x_i = (shapeInf[0 * 5 + 1] <= 1) ? 0 : i;
                                int index_x_j = (shapeInf[0 * 5 + 2] <= 1) ? 0 : j;
                                int index_x_k = (shapeInf[0 * 5 + 3] <= 1) ? 0 : k;
                                int index_x_l = (shapeInf[0 * 5 + 4] <= 1) ? 0 : l;

                                TYPE_Y x1_value = static_cast<TYPE_Y>(Gm_x(index_x_i * shapeInf[0 * 5 + 2] * shapeInf[0 * 5 + 3] * shapeInf[0 * 5 + 4] + index_x_j * shapeInf[0 * 5 + 3] * shapeInf[0 * 5 + 4] + index_x_k * shapeInf[0 * 5 + 4] + index_x_l));
                                Duplicate<TYPE_Y>(tmp1, static_cast<TYPE_Y>(x1_value), 1);
                                Exp(tmp1, tmp1, 1);
                                Div(tmp1, tmp1, tmp2, 1);
                                Gm_y(i * max_index[1] * max_index[2] * max_index[3] + j * max_index[2] * max_index[3] + k * max_index[3] + l) = static_cast<TYPE_Y>(tmp1(0));
                            }
                            Duplicate<TYPE_Y>(tmp2, static_cast<TYPE_Y>(0), 1); 
                        }
                    }
                }
            }
            else if (attrdim == 1 || attrdim == -3) {
                for (int i = 0; i < max_index[0]; i++) {
                    for (int k = 0; k < max_index[2]; k++) {
                        for (int l = 0; l < max_index[3]; l++) {
                            float sum = 0;
                            for (int j = 0; j < max_index[1]; j++) {
                                int index_x_i = (shapeInf[0 * 5 + 1] <= 1) ? 0 : i;
                                int index_x_j = (shapeInf[0 * 5 + 2] <= 1) ? 0 : j;
                                int index_x_k = (shapeInf[0 * 5 + 3] <= 1) ? 0 : k;
                                int index_x_l = (shapeInf[0 * 5 + 4] <= 1) ? 0 : l;
                                TYPE_Y x1_value = static_cast<TYPE_Y>(Gm_x(index_x_i * shapeInf[0 * 5 + 2] * shapeInf[0 * 5 + 3] * shapeInf[0 * 5 + 4] + index_x_j * shapeInf[0 * 5 + 3] * shapeInf[0 * 5 + 4] + index_x_k * shapeInf[0 * 5 + 4] + index_x_l));
                                Duplicate<TYPE_Y>(tmp1, static_cast<TYPE_Y>(x1_value), 1);
                                Exp(tmp1, tmp1, 1);
                                Adds(tmp2, tmp2, static_cast<TYPE_Y>(tmp1(0)), 1); 
                            }

                            for (int j = 0; j < max_index[1]; j++) {
                                int index_x_i = (shapeInf[0 * 5 + 1] <= 1) ? 0 : i;
                                int index_x_j = (shapeInf[0 * 5 + 2] <= 1) ? 0 : j;
                                int index_x_k = (shapeInf[0 * 5 + 3] <= 1) ? 0 : k;
                                int index_x_l = (shapeInf[0 * 5 + 4] <= 1) ? 0 : l;

                                TYPE_Y x1_value = static_cast<TYPE_Y>(Gm_x(index_x_i * shapeInf[0 * 5 + 2] * shapeInf[0 * 5 + 3] * shapeInf[0 * 5 + 4] + index_x_j * shapeInf[0 * 5 + 3] * shapeInf[0 * 5 + 4] + index_x_k * shapeInf[0 * 5 + 4] + index_x_l));
                                Duplicate<TYPE_Y>(tmp1, static_cast<TYPE_Y>(x1_value), 1);
                                Exp(tmp1, tmp1, 1);
                                Div(tmp1, tmp1, tmp2, 1);
                                Gm_y(i * max_index[1] * max_index[2] * max_index[3] + j * max_index[2] * max_index[3] + k * max_index[3] + l) = static_cast<TYPE_Y>(tmp1(0));
                            }
                            Duplicate<TYPE_Y>(tmp2, static_cast<TYPE_Y>(0), 1);  
                        }
                    }
                }
            }
            else if (attrdim == 2 || attrdim == -2) {
                for (int i = 0; i < max_index[0]; i++) {
                    for (int j = 0; j < max_index[1]; j++) {
                        for (int l = 0; l < max_index[3]; l++) {
                            float sum = 0;
                            for (int k = 0; k < max_index[2]; k++) {
                                int index_x_i = (shapeInf[0 * 5 + 1] <= 1) ? 0 : i;
                                int index_x_j = (shapeInf[0 * 5 + 2] <= 1) ? 0 : j;
                                int index_x_k = (shapeInf[0 * 5 + 3] <= 1) ? 0 : k;
                                int index_x_l = (shapeInf[0 * 5 + 4] <= 1) ? 0 : l;

                                TYPE_Y x1_value = static_cast<TYPE_Y>(Gm_x(index_x_i * shapeInf[0 * 5 + 2] * shapeInf[0 * 5 + 3] * shapeInf[0 * 5 + 4] + index_x_j * shapeInf[0 * 5 + 3] * shapeInf[0 * 5 + 4] + index_x_k * shapeInf[0 * 5 + 4] + index_x_l));
                                Duplicate<TYPE_Y>(tmp1, static_cast<TYPE_Y>(x1_value), 1);
                                Exp(tmp1, tmp1, 1);
                                Adds(tmp2, tmp2, static_cast<TYPE_Y>(tmp1(0)), 1); 
                            }

                            for (int k = 0; k < max_index[2]; k++) {
                                int index_x_i = (shapeInf[0 * 5 + 1] <= 1) ? 0 : i;
                                int index_x_j = (shapeInf[0 * 5 + 2] <= 1) ? 0 : j;
                                int index_x_k = (shapeInf[0 * 5 + 3] <= 1) ? 0 : k;
                                int index_x_l = (shapeInf[0 * 5 + 4] <= 1) ? 0 : l;

                                TYPE_Y x1_value = static_cast<TYPE_Y>(Gm_x(index_x_i * shapeInf[0 * 5 + 2] * shapeInf[0 * 5 + 3] * shapeInf[0 * 5 + 4] + index_x_j * shapeInf[0 * 5 + 3] * shapeInf[0 * 5 + 4] + index_x_k * shapeInf[0 * 5 + 4] + index_x_l));
                                Duplicate<TYPE_Y>(tmp1, static_cast<TYPE_Y>(x1_value), 1);
                                Exp(tmp1, tmp1, 1);
                                Div(tmp1, tmp1, tmp2, 1);
                                Gm_y(i * max_index[1] * max_index[2] * max_index[3] + j * max_index[2] * max_index[3] + k * max_index[3] + l) = static_cast<TYPE_Y>(tmp1(0));
                            }
                            Duplicate<TYPE_Y>(tmp2, static_cast<TYPE_Y>(0), 1); 
                        }
                    }
                }
            }
            else if (attrdim == 3 || attrdim == -1) { 
                for (int i = 0; i < max_index[0]; i++) {
                    for (int j = 0; j < max_index[1]; j++) {
                        for (int k = 0; k < max_index[2]; k++) {
                            float sum = 0;
                            for (int l = 0; l < max_index[3]; l++) {
                                int index_x_i = (shapeInf[0 * 5 + 1] <= 1) ? 0 : i;
                                int index_x_j = (shapeInf[0 * 5 + 2] <= 1) ? 0 : j;
                                int index_x_k = (shapeInf[0 * 5 + 3] <= 1) ? 0 : k;
                                int index_x_l = (shapeInf[0 * 5 + 4] <= 1) ? 0 : l;

                                TYPE_Y x1_value = static_cast<TYPE_Y>(Gm_x(index_x_i * shapeInf[0 * 5 + 2] * shapeInf[0 * 5 + 3] * shapeInf[0 * 5 + 4] + index_x_j * shapeInf[0 * 5 + 3] * shapeInf[0 * 5 + 4] + index_x_k * shapeInf[0 * 5 + 4] + index_x_l));
                                Duplicate<TYPE_Y>(tmp1, static_cast<TYPE_Y>(x1_value), 1);
                                Exp(tmp1, tmp1, 1);
                                Adds(tmp2, tmp2, static_cast<TYPE_Y>(tmp1(0)), 1); 
                            }

                            for (int l = 0; l < max_index[3]; l++) {
                                int index_x_i = (shapeInf[0 * 5 + 1] <= 1) ? 0 : i;
                                int index_x_j = (shapeInf[0 * 5 + 2] <= 1) ? 0 : j;
                                int index_x_k = (shapeInf[0 * 5 + 3] <= 1) ? 0 : k;
                                int index_x_l = (shapeInf[0 * 5 + 4] <= 1) ? 0 : l;

                                TYPE_Y x1_value = static_cast<TYPE_Y>(Gm_x(index_x_i * shapeInf[0 * 5 + 2] * shapeInf[0 * 5 + 3] * shapeInf[0 * 5 + 4] + index_x_j * shapeInf[0 * 5 + 3] * shapeInf[0 * 5 + 4] + index_x_k * shapeInf[0 * 5 + 4] + index_x_l));
                                Duplicate<TYPE_Y>(tmp1, static_cast<TYPE_Y>(x1_value), 1);
                                Exp(tmp1, tmp1, 1);
                                Div(tmp1, tmp1, tmp2, 1);
                                Gm_y(i * max_index[1] * max_index[2] * max_index[3] + j * max_index[2] * max_index[3] + k * max_index[3] + l) = static_cast<TYPE_Y>(tmp1(0));
                            }
                            Duplicate<TYPE_Y>(tmp2, static_cast<TYPE_Y>(0), 1); 
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
    TBuf<QuePosition::VECCALC> tmp1Buffer,tmp2Buffer,tmp32_1Buffer,tmp32_2Buffer; 
    uint32_t blockLength;    
};  
     
template<typename TYPE_X,  typename TYPE_Y> class KernelSoftmax_Fast {    
    using T = TYPE_Y;            
public:                              
    __aicore__ inline KernelSoftmax_Fast() {}         
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, TPipe* pipeIn) {                         
        Gm_x.SetGlobalBuffer((__gm__ half*)x , BlockLength);                 
        Gm_y.SetGlobalBuffer((__gm__ half*)y , BlockLength);           
        pipe = pipeIn;  
        pipe->InitBuffer(Q_x, BUFFER_NUM, TileLength * 2);       
        pipe->InitBuffer(Q_y, BUFFER_NUM, TileLength * 2);   
        pipe->InitBuffer(tmpMBuffer, TileLength * 4);            
        pipe->InitBuffer(tmpXBuffer, TileLength * 4);    
    }   
    __aicore__ inline void Process() {      
        int32_t i;      
        for (i = 0; i < TileNum; i++) {         
            CopyIn(i,  TileLength);   
            Compute(i, TileLength);       
            CopyOut(i, TileLength);       
        }      
    }       
private:
    __aicore__ inline void CopyIn(int32_t progress, uint32_t length) {  
        LocalTensor<half>  x = Q_x.AllocTensor<half>();    
        DataCopy(x, Gm_x[progress * TileLength], length);    
        Q_x.EnQue(x);     
    }    
    __aicore__ inline void Compute(int32_t progress, uint32_t length) { 
        LocalTensor<half> x     = Q_x.DeQue<half>();          
        LocalTensor<half> y     = Q_y.AllocTensor<half>();           
        LocalTensor<float> tmpM  = tmpMBuffer.Get<float>();                     
        LocalTensor<float> tmpX  = tmpXBuffer.Get<float>();     
        Cast(tmpX, x, RoundMode::CAST_NONE, TileLength);            
        Exp(tmpX, tmpX, TileLength);        
        if(progress==0)ReduceSum(tmpM, tmpX, tmpM, 480);                    
        Muls(tmpX,tmpX,1/tmpM(0),TileLength);       
        Cast(y, tmpX, RoundMode::CAST_ROUND, TileLength);           
        Q_x.FreeTensor(x);    
        Q_y.EnQue<half>(y);        
    }            
    __aicore__ inline void CopyOut(int32_t progress, uint32_t length) {
        LocalTensor<half> y = Q_y.DeQue<half>();
        DataCopy(Gm_y[progress * TileLength], y, length);  
        Q_y.FreeTensor(y);
    }    
 
private:  
    TPipe* pipe;  
    TQue<QuePosition::VECIN, BUFFER_NUM> Q_x;               
    TQue<QuePosition::VECOUT, BUFFER_NUM> Q_y;
    TBuf<QuePosition::VECCALC> tmpMBuffer, tmpXBuffer;     
    GlobalTensor<half> Gm_x;              
    GlobalTensor<half> Gm_y;   
    uint32_t index;     
};        
 
extern "C" __global__ __aicore__ void softmax(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);     
    if(TILING_KEY_IS(1)){     
        TPipe pipe;         
        KernelSoftmax_Fast<DTYPE_X, DTYPE_Y> op;         
        op.Init(x, y,&pipe);          
        op.Process();       
    } 
    else if(TILING_KEY_IS(2)){              
        KernelSoftmax_Norm<DTYPE_X, DTYPE_Y> op;             
        op.Init(x, y, tiling_data.core_size);     
        op.Process(tiling_data.shapeInf, tiling_data.attrdim);         
    } 
}   
    

 
 
  
