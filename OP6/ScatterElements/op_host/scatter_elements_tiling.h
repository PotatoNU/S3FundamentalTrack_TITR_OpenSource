 
#include "register/tilingdata_base.h"
  
namespace optiling {   
BEGIN_TILING_DATA_DEF(TilingData)
  TILING_DATA_FIELD_DEF(uint32_t, ALIGN_NUM);  
  TILING_DATA_FIELD_DEF(uint32_t, core_size); 
  TILING_DATA_FIELD_DEF(uint32_t, core_remain);    
  TILING_DATA_FIELD_DEF(uint32_t, F_length);    
  TILING_DATA_FIELD_DEF_ARR(uint32_t, 10, shapeInf);         
  TILING_DATA_FIELD_DEF(int, attrAxis);      
  TILING_DATA_FIELD_DEF(uint8_t, mode);                       
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(ScatterElements, TilingData)
}