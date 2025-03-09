 
#include "register/tilingdata_base.h"
namespace optiling {     
BEGIN_TILING_DATA_DEF(TilingData)    
  TILING_DATA_FIELD_DEF(uint32_t, core_size);      
  TILING_DATA_FIELD_DEF(uint16_t, block_size);     
  TILING_DATA_FIELD_DEF(uint16_t, tileNum);               
END_TILING_DATA_DEF;        
   
REGISTER_TILING_DATA_CLASS(AsinhGrad, TilingData) 
}   