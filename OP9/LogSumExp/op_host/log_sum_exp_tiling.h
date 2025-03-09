
#include "register/tilingdata_base.h"

namespace optiling
{
  BEGIN_TILING_DATA_DEF(TilingData)
  TILING_DATA_FIELD_DEF(uint32_t, ALIGN_NUM);
  TILING_DATA_FIELD_DEF(uint32_t, core_size);   
  TILING_DATA_FIELD_DEF(uint32_t, core_remain);
  TILING_DATA_FIELD_DEF_ARR(uint32_t, 5, shapeInf);  
  TILING_DATA_FIELD_DEF_ARR(int, 5, attrdim);         
  END_TILING_DATA_DEF;

  REGISTER_TILING_DATA_CLASS(LogSumExp, TilingData)
}