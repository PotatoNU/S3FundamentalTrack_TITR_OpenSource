
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(IsCloseTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, block_size);   
  TILING_DATA_FIELD_DEF(uint32_t, core_size);   
  TILING_DATA_FIELD_DEF(uint32_t, core_remain);     
  TILING_DATA_FIELD_DEF_ARR(uint32_t, 10, shapeInf);          
  TILING_DATA_FIELD_DEF(uint8_t, ALIGN_NUM);   
  TILING_DATA_FIELD_DEF(bool, isTs);           
  TILING_DATA_FIELD_DEF(float, rtol);     
  TILING_DATA_FIELD_DEF(float, atol);       
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(IsClose, IsCloseTilingData)
}
