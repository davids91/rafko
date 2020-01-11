#ifndef sparse_net_global_H
#define sparse_net_global_H

namespace sparse_net_library {
  
/**
 * GLOBAL TYPES
 */
typedef unsigned long uint64;
typedef unsigned int uint32;
typedef unsigned short uint16;
typedef unsigned char uint8;
typedef signed long sint64;
typedef signed int sint32;
typedef signed short sint16;
typedef signed char sint8;
typedef double sdouble32;
typedef uint16* p_uint16;
typedef sdouble32* p_sdouble32;

/**
 * GLOBAL PARAMETERS
 *//* Later to be migrated to the server implementation.. */
extern sdouble32 epsilon; /* very small positive value almost greater, than 0.0 */
extern sdouble32 lambda;
extern sdouble32 alpha;

} /* namespace sparse_net_library */
#endif /* defined sparse_net_global_H */
