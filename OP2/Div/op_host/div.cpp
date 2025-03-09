 
#include "div_tiling.h" 
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"                                                                        
namespace optiling {                              
const uint32_t BLOCK_SIZE = 32;             
static ge::graphStatus TilingFunc(gert::TilingContext* context) {        
    TilingData tiling;  
    int32_t NUM = 8;   
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint64_t ub_size; ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);  
    auto aivNum = ascendcPlatform.GetCoreNum();
    uint32_t input_num=2;                  
    uint32_t shapeInf[8] = {};  
    uint32_t inputLength[2] = {};   
    uint32_t length = 0;        
    for (int i = 0; i < input_num; ++i) 
        length = std::max<uint32_t>(length, context->GetInputShape(i)->GetStorageShape().GetDimNum());
    for (int i = 0; i < input_num; ++i) { 
        const gert::StorageShape* shape = context->GetInputShape(i);
        inputLength[i] = context->GetInputTensor(i)->GetShapeSize();       
        shapeInf[i*4]=shape->GetStorageShape().GetDimNum();           
        for (int j = 1; j <= shape->GetStorageShape().GetDimNum(); j++) {    
            shapeInf[i*4+j] = shape->GetStorageShape().GetDim(j-1);                   
        } 
    }               
    uint32_t total_length = 0;
    for (int i = 0; i < input_num; ++i) {  
        total_length = std::max<uint32_t>(total_length, context->GetInputTensor(i)->GetShapeSize());
    }             
    bool boardCast = 0;    
    if (inputLength[0] != total_length ||           
        inputLength[1] != total_length ) {    
        boardCast = 1;  
    }      
    bool isTs=false;  
    if(inputLength[0]>500000 ||    
       inputLength[1]>500000 ){           
        isTs=true;          
        context->SetTilingKey(0);              
    }else{
        context->SetTilingKey(1);       
    }
    auto dt = context->GetInputTensor(0)->GetDataType();
    uint32_t sizeofdatatype;  
    if (dt == ge::DT_INT8) {      
        sizeofdatatype = 1;  
        NUM =  8;    
    }else if(dt == ge::DT_INT32) {
        sizeofdatatype = 4;
        NUM = 3*2+2;       
    } 
    else if (dt == ge::DT_FLOAT16 ) { 
        sizeofdatatype = 2;
        NUM = 6;
    }         
    else {      
        sizeofdatatype = 4;
        NUM = 6;               
    }               
    uint32_t ALIGN_NUM = BLOCK_SIZE / sizeofdatatype;                   
    uint32_t tiling_size = ((ub_size) / BLOCK_SIZE / 1) / NUM;          
    tiling_size = tiling_size <= 8 ? tiling_size : tiling_size / 8 * 8; 
    uint32_t block_size = tiling_size * ALIGN_NUM;
    aivNum = 1;   
    uint32_t core_size = (total_length / aivNum) / (ALIGN_NUM * 8) * (ALIGN_NUM * 8);
    uint32_t core_remain = total_length - aivNum * core_size;
    tiling.set_ALIGN_NUM(ALIGN_NUM);        
    tiling.set_block_size(block_size);            
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
class Div : public OpDef {
public:
    explicit Div(const char* name) : OpDef(name)
    {
        this->Input("x1")   
            .ParamType(REQUIRED) 
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,  ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        this->Input("x2")     
            .ParamType(REQUIRED) 
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,  ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        
        this->Output("y")  
            .ParamType(REQUIRED)  
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,  ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
      
        this->SetInferShape(ge::InferShape);   

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend310b");
    }
};
 
OP_ADD(Div); 
}
  
