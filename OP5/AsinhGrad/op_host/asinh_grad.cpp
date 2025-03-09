 
#include "asinh_grad._tiling.h"    
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"        
namespace optiling {                                
const uint32_t BLOCK_SIZE = 32;          
static ge::graphStatus TilingFunc(gert::TilingContext* context) {    
    TilingData tiling;     
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint64_t ub_size; ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);  
    int32_t NUM = 0;   
    auto dt = context->GetInputTensor(0)->GetDataType();   
    uint32_t sizeofdatatype;  
    if (dt == ge::DT_FLOAT16) { 
        sizeofdatatype = 2;
        NUM = 5;   
    }             
    else {            
        sizeofdatatype = 4;   
        NUM = 3;     
    }                  
    uint32_t input_num=1;                  
    uint32_t shapeInf[1 * 10] = {};  
    uint32_t inputLength[1] = {};      
    uint32_t length = 0;      
    for (int i = 0; i < input_num; ++i)   
        length = std::max<uint32_t>(length, context->GetInputShape(i)->GetStorageShape().GetDimNum());
    for (int i = 0; i < input_num; ++i) { 
        const gert::StorageShape* shape = context->GetInputShape(i);
        inputLength[i] = context->GetInputTensor(i)->GetShapeSize();       
        shapeInf[i*10+0]=shape->GetStorageShape().GetDimNum();            
        for (int j = 1; j <= shape->GetStorageShape().GetDimNum(); j++) {  
            shapeInf[i*10+j] = shape->GetStorageShape().GetDim(j-1);                     
        } 
    }                        
    if(dt == ge::DT_FLOAT16 && shapeInf[0]==3 && inputLength[0]>=1024*1024){                                             
        context->SetTilingKey(1);                          
        NUM = 2*1+2;                                                                              
    }else{                    
        context->SetTilingKey(0);             
    }         
    uint32_t total_length = 0;
    for (int i = 0; i < input_num; ++i) {   
        total_length = std::max<uint32_t>(total_length, context->GetInputTensor(i)->GetShapeSize());
    }       
    uint32_t ALIGN_NUM = BLOCK_SIZE / sizeofdatatype;                   
    uint32_t tiling_size = ((ub_size) / BLOCK_SIZE / 2) / NUM;          
    tiling_size = tiling_size <= 8 ? tiling_size : tiling_size / 8 * 8;
    uint16_t block_size = tiling_size * ALIGN_NUM;   
    uint32_t core_size = (total_length) / (ALIGN_NUM * 8) * (ALIGN_NUM * 8);
    uint32_t core_remain = total_length - core_size;                   
    core_size = core_size + core_remain;  
    core_size = core_size + (core_size % ALIGN_NUM ? ALIGN_NUM - core_size % ALIGN_NUM : 0);                   
    if(dt == ge::DT_FLOAT16 && shapeInf[0]==3 && inputLength[0]>=1024*1024){                                                           
        block_size = block_size+32*16;                                                                                       
    }             
    uint16_t tileNum = core_size / block_size + (core_size % block_size > 0)-1;                      
    tiling.set_tileNum(tileNum);           
    tiling.set_block_size(block_size);     
    tiling.set_core_size(core_size);      
    context->SetBlockDim(1);   
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    size_t usrSize = 0;       
    uint32_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize(); 
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
class AsinhGrad : public OpDef {
public:
    explicit AsinhGrad(const char* name) : OpDef(name)
    {
        this->Input("y")    
            .ParamType(REQUIRED) 
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});

        this->Input("dy")    
            .ParamType(REQUIRED) 
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})   
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});

        this->Output("z")      
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
    
OP_ADD(AsinhGrad); 
} 
