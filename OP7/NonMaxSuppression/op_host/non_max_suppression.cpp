 
#include "non_max_suppression_tiling.h" 
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#define MAX_NUM_BOXES 1024   
namespace optiling {                            
const uint32_t BLOCK_SIZE = 32;     
static ge::graphStatus TilingFunc(gert::TilingContext* context) {   
    TilingData tiling;  
    int32_t NUM = 6;     
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint64_t ub_size; ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);  
    auto aivNum = ascendcPlatform.GetCoreNum(); 
    int center_point_box = *context->GetAttrs()->GetInt(0);    
    tiling.set_center_point_box(center_point_box);        
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
    auto dt = context->GetInputTensor(0)->GetDataType();
    uint32_t sizeofdatatype;  
    if (dt == ge::DT_INT8) {
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
    size_t usrSize = 7*1024*4;                    
    uint32_t sysWorkspaceSize = 0;           
    size_t *currentWorkspace = context->GetWorkspaceSizes(1); 
    currentWorkspace[0] = usrSize + sysWorkspaceSize;        
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
class NonMaxSuppression : public OpDef {
public:
    explicit NonMaxSuppression(const char* name) : OpDef(name)
    {
        this->Input("boxes")   
            .ParamType(REQUIRED) 
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->Input("scores")   
            .ParamType(REQUIRED) 
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->Input("max_output_boxes_per_class")   
            .ParamType(REQUIRED) 
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND}) 
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->Input("iou_threshold")   
            .ParamType(REQUIRED) 
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->Input("score_threshold")   
            .ParamType(REQUIRED) 
            .DataType({ge::DT_FLOAT})   
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->Output("selected_indices")  
            .ParamType(REQUIRED)  
            .DataType({ge::DT_INT32})        
            .Format({ge::FORMAT_ND})  
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->Attr("center_point_box").AttrType(OPTIONAL).Int(0);     
      
        this->SetInferShape(ge::InferShape);   

        this->AICore()
            .SetTiling(optiling::TilingFunc);    
        this->AICore().AddConfig("ascend310b");

    }
};
 
OP_ADD(NonMaxSuppression);  
}
  
