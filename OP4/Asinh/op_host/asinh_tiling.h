 
#include "register/tilingdata_base.h"  
namespace optiling {    
BEGIN_TILING_DATA_DEF(TilingData)  
  TILING_DATA_FIELD_DEF(uint32_t, block_size);   
  TILING_DATA_FIELD_DEF(uint32_t, core_size);      
  TILING_DATA_FIELD_DEF(uint32_t, core_remain);       
  TILING_DATA_FIELD_DEF(bool, isTs);             
END_TILING_DATA_DEF; 
REGISTER_TILING_DATA_CLASS(Asinh, TilingData)    
}           