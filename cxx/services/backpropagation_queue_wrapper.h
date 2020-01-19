#ifndef BACKPROPAGATION_WRAPPER_H
#define BACKPROPAGATION_WRAPPER_H

#include "sparse_net_global.h"
#include "gen/sparse_net.pb.h"
#include "gen/solution.pb.h"
#include "models/service_context.h"

namespace sparse_net_library{

/**
 * @brief      Wrapper function to generate Backpropagation_queue objects from @SparseNet
 *             objects
 */
class Backpropagation_queue_wrapper{
public:
  Backpropagation_queue_wrapper(SparseNet& net, Service_context context = Service_context());
  Backpropagation_queue operator()(){
    return gradient_step;
  }
private:
  Backpropagation_queue gradient_step;
};

} /* namespace sparse_net_library */

#endif /* BACKPROPAGATION_WRAPPER_H */