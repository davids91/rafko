#ifndef weight_initializer_H
#define weight_initializer_H


#include "models/gen/sparse_net.pb.h"

#include "sparse_net_global.h"

namespace sparse_net_library {

class Weight_initializer
{
public:
  Weight_initializer() noexcept;
  virtual sdouble32 nextWeightFor(transfer_functions usedTransferFnc) const = 0;
  virtual sdouble32 nextMemRatio() const = 0;
  virtual sdouble32 nextWeight() const = 0;
  virtual sdouble32 nextBias() const = 0;

  void set(uint32 expInputNumber, sdouble32 expInputMax);
  ~Weight_initializer();
protected:
  uint32 expInNum = 0;
  sdouble32 expInMax = 0.0;
};

} /* namespace sparse_net_library */
#endif /* weight_initializer_H */
