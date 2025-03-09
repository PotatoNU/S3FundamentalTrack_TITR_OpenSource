#include "scatter_elements_tiling.h" 
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include <algorithm>   
#include <vector>
#include <cstring> 
#include <cstdlib>       
namespace optiling {                                
const uint32_t BLOCK_SIZE = 32;     
static ge::graphStatus TilingFunc(gert::TilingContext* context) {   
    TilingData tiling;  
    int32_t NUM = 6;            
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint64_t ub_size; ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);  
    auto aivNum = ascendcPlatform.GetCoreNum();                 
    int attrAxis = *context->GetAttrs()->GetInt(0);    
    tiling.set_attrAxis(attrAxis);     
    uint8_t mode=0; 
    const char *str = context->GetAttrs()->GetAttrPointer<char>(1); 
    if (strcmp(str, "add") == 0) {
        mode=1;  
    }else if(strcmp(str, "multiply") == 0) {
        mode=2;  
    }else { 
        mode=3;     
    } 
    tiling.set_mode(mode);   
    uint32_t input_num=2;                  
    uint32_t shapeInf[2 * 5] = {};  
    uint32_t inputLength[2] = {};    
    uint32_t length = 0;          
    for (int i = 0; i < input_num; ++i) 
        length = std::max<uint32_t>(length, context->GetInputShape(i)->GetStorageShape().GetDimNum());
    for (int i = 0; i < input_num; ++i) { 
        const gert::StorageShape* shape = context->GetInputShape(i);
        inputLength[i] = context->GetInputTensor(i)->GetShapeSize();       
        shapeInf[i*5+0]=shape->GetStorageShape().GetDimNum();            
        for (int j = 1; j <= shape->GetStorageShape().GetDimNum(); j++) {   
            shapeInf[i*5+j] = shape->GetStorageShape().GetDim(j-1);                   
        } 
    }              
    uint32_t total_length = 0;
    for (int i = 0; i < input_num; ++i) {  
        total_length = std::max<uint32_t>(total_length, context->GetInputTensor(i)->GetShapeSize());
    }       
    bool boardCast = 0;    
    if (inputLength[0] != total_length ||       
        inputLength[1] != total_length ||
        inputLength[2] != total_length) {    
        boardCast = 1;  
    }           
    auto dt = context->GetInputTensor(0)->GetDataType();
    int F_length = 1;
    for (int i = 1; i <= shapeInf[1*5+0]; ++i) { 
        F_length *= shapeInf[1*5+i];     
    }          
    tiling.set_F_length(F_length);            
    bool IsC5=false;  
    if(mode%3==0 && dt == ge::DT_FLOAT && length==3 && attrAxis==0){ 
        context->SetTilingKey(0);                 
        IsC5 =true;   
        auto Gm_indices  = context->GetInputTensor(1)->GetData<int32_t>();  
        uint32_t c5result[1024*3] = {};                
        uint32_t upto;
        if(F_length<=1024*3)upto = F_length;
        else upto = 1024*3;       
        for (int i = 0; i < upto; ++i) {                     
            c5result[i] = Gm_indices[i] * (shapeInf[0*5+2] * shapeInf[0*5+3]) + 
                         static_cast<int>((i / shapeInf[1 * 5 + 3]) % shapeInf[1 * 5 + 2]) * shapeInf[0*5+3] + 
                         static_cast<int>(i % shapeInf[1 * 5 + 3]);  
        }   
    }else context->SetTilingKey(1);             
    uint32_t sizeofdatatype;  
    if (dt == ge::DT_UINT8) {  
        sizeofdatatype = 1;
        NUM = 12;
    }
    else if (dt == ge::DT_FLOAT16 || dt == ge::DT_BF16) { 
        sizeofdatatype = 2; 
    }         
    else {
        sizeofdatatype = 4; 
    }              
    uint32_t ALIGN_NUM = BLOCK_SIZE / sizeofdatatype;                   
    uint32_t tiling_size = ((ub_size) / BLOCK_SIZE / 2) / NUM;         
    tiling_size = tiling_size <= 8 ? tiling_size : tiling_size / 8 * 8;
    uint32_t block_size = tiling_size * ALIGN_NUM;
    aivNum = (aivNum < total_length / block_size) ? aivNum : (total_length / block_size);
    aivNum = aivNum >= 1 ? aivNum : 1;      
    uint32_t core_size = (total_length / aivNum) / (ALIGN_NUM * 8) * (ALIGN_NUM * 8);
    uint32_t core_remain = total_length - aivNum * core_size;
    tiling.set_ALIGN_NUM(ALIGN_NUM);         
    tiling.set_core_size(core_size); 
    tiling.set_core_remain(core_remain);  
    tiling.set_shapeInf(shapeInf);      
    context->SetBlockDim(aivNum);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}                   
} 
 
namespace ge { 
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* x1_shape = context->GetInputShape(0);
    gert::Shape* y_shape = context->GetOutputShape(0);
    *y_shape = *x1_shape;
    return GRAPH_SUCCESS;
}
}

namespace ops {
class ScatterElements : public OpDef {
public:
    explicit ScatterElements(const char* name) : OpDef(name)
    {
        this->Input("var")  
            .ParamType(REQUIRED)   
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_UINT8}) 
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("indices")  
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})      
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("updates")  
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_UINT8})   
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        this->Attr("axis").AttrType(OPTIONAL).Int(0);               
        this->Attr("reduce").AttrType(OPTIONAL).String("None");                

        this->SetInferShape(ge::InferShape);    
        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend310b");

    } 
};
 
OP_ADD(ScatterElements); 
}
  
  