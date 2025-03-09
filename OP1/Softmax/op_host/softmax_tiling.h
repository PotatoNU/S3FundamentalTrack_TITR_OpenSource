 
#include "register/tilingdata_base.h"
namespace optiling {     
BEGIN_TILING_DATA_DEF(TilingData) 
  TILING_DATA_FIELD_DEF(uint32_t, core_size);          
  TILING_DATA_FIELD_DEF_ARR(uint32_t, 5, shapeInf);       
  TILING_DATA_FIELD_DEF(int, attrdim);                           
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Softmax, TilingData)
} 