 
#include "softmax_tiling.h" 
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"    
#define Debug   
namespace optiling {                              
const uint32_t BLOCK_SIZE = 32;     
static ge::graphStatus TilingFunc(gert::TilingContext* context) {   
    TilingData tiling;   
    int32_t NUM = 8;      
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint64_t ub_size; ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);  
    auto aivNum = ascendcPlatform.GetCoreNum();           
    uint32_t input_num=1;                  
    uint32_t shapeInf[5] = {};  
    uint32_t inputLength[1] = {};    
    uint32_t length = 0;        
    for (int i = 0; i < input_num; ++i) 
        length = std::max<uint32_t>(length, context->GetInputShape(i)->GetStorageShape().GetDimNum());
    for (int i = 0; i < input_num; ++i) { 
        const gert::StorageShape* shape = context->GetInputShape(i);
        inputLength[i] = context->GetInputTensor(i)->GetShapeSize();       
        shapeInf[i*5]=shape->GetStorageShape().GetDimNum();             
        for (int j = 1; j <= shape->GetStorageShape().GetDimNum(); j++) {  
            shapeInf[i*5+j] = shape->GetStorageShape().GetDim(j-1);                   
        } 
    }        
    int attrdim = *context->GetAttrs()->GetInt(0);    
    if(attrdim<0){ 
        attrdim+=shapeInf[0];     
    }  
    tiling.set_attrdim((attrdim));        
    uint32_t total_length = 0;
    for (int i = 0; i < input_num; ++i) {  
        total_length = std::max<uint32_t>(total_length, context->GetInputTensor(i)->GetShapeSize());
    }            
    if(attrdim == 2 && shapeInf[0]%3==0 && shapeInf[1]%16==0 &&      
       shapeInf[2]%128==0 && shapeInf[3]%480==0 ){  
        context->SetTilingKey(1);                          
    }else{  
        context->SetTilingKey(2);                     
    }   
    auto dt = context->GetInputTensor(0)->GetDataType();
    uint32_t sizeofdatatype;  
    if (dt == ge::DT_FLOAT16) {  
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
    core_size = core_size + core_remain;       
    core_size = core_size + (core_size % ALIGN_NUM ? ALIGN_NUM - core_size % ALIGN_NUM : 0); 
    tiling.set_core_size(core_size);        
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
class Softmax : public OpDef {
public:
    explicit Softmax(const char* name) : OpDef(name)
    {
        this->Input("x")        
            .ParamType(REQUIRED) 
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND}); 

        this->Output("y")  
            .ParamType(REQUIRED)  
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND}); 

        this->Attr("dim").AttrType(OPTIONAL).Int(-1);  
      
        this->SetInferShape(ge::InferShape);   

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend310b");

    }
};  
  
OP_ADD(Softmax); 
}
  
