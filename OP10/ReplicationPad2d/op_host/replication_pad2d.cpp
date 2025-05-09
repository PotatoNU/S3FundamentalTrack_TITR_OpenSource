 
#include "replication_pad2d_tiling.h"      
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"        
namespace optiling {                            
const uint32_t BLOCK_SIZE = 32;     
static ge::graphStatus TilingFunc(gert::TilingContext* context) {   
    TilingData tiling;    
    int32_t NUM = 9;  
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint64_t ub_size; ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);  
    auto aivNum = ascendcPlatform.GetCoreNum();   
    uint32_t input_num=1;                    
    uint32_t shapeInf[1 * 5] = {};  
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
        inputLength[1] != total_length ) {    
        boardCast = 1;  
    }      
    auto dt = context->GetInputTensor(0)->GetDataType();
    uint32_t sizeofdatatype;  
    if (dt == ge::DT_INT8) {
        sizeofdatatype = 1;
        NUM = 12;
    }
    else if (dt == ge::DT_FLOAT16 || dt == ge::DT_BF16) { 
        sizeofdatatype = 2;
        tiling.set_dtx(16);           
    }         
    else {
        sizeofdatatype = 4;
        tiling.set_dtx(32);        
    }               
    uint32_t ALIGN_NUM = BLOCK_SIZE / sizeofdatatype;                   
    uint32_t tiling_size = ((ub_size) / BLOCK_SIZE / 2) / NUM;          
    tiling_size = tiling_size <= 8 ? tiling_size : tiling_size / 8 * 8; 
    uint32_t block_size = tiling_size * ALIGN_NUM;
    uint32_t core_size = (total_length / 1) / (ALIGN_NUM * 8) * (ALIGN_NUM * 8);
    uint32_t core_remain = total_length - 1 * core_size;   
    tiling.set_ALIGN_NUM(ALIGN_NUM);     
    tiling.set_block_size(block_size);  
    tiling.set_core_size(core_size);    
    tiling.set_core_remain(core_remain);   
    tiling.set_shapeInf(shapeInf);          
    context->SetBlockDim(1);
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
class ReplicationPad2d : public OpDef { 
public:
    explicit ReplicationPad2d(const char* name) : OpDef(name)
    {
        this->Input("x")   
            .ParamType(REQUIRED) 
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});

        this->Input("paddings")    
            .ParamType(REQUIRED) 
            .DataType({ge::DT_INT32, ge::DT_INT32})      
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})      
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});

        this->Output("y")      
            .ParamType(REQUIRED)   
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16}) 
            .Format({ge::FORMAT_ND, ge::FORMAT_ND}) 
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape);          

        this->AICore()
            .SetTiling(optiling::TilingFunc); 
        this->AICore().AddConfig("ascend310b");

    }
};
 
OP_ADD(ReplicationPad2d); 
} 
   
    