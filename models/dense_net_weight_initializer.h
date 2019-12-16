#ifndef Dense_net_weight_initializer_H
#define Dense_net_weight_initializer_H

#include "models/weight_initializer.h"

namespace sparse_net_library {

class Dense_net_weight_initializer : public Weight_initializer
{
public:
  Dense_net_weight_initializer();
  Dense_net_weight_initializer(uint32 seed, sdouble32 memRatioMin = 0.0, sdouble32 memRatioMax = 1.0);
  void set(uint32 expInputNumber, sdouble32 expInputMax);
  sdouble32 nextWeight() const;
  sdouble32 nextWeightFor(transfer_functions usedTransferFnc) const;
  sdouble32 nextMemRatio() const;
  sdouble32 nextBias() const;
  ~Dense_net_weight_initializer();

private:
  sdouble32 memMin = 0.0;
  sdouble32 memMax = 1.0;

  sdouble32 getWeightAmp(transfer_functions usedTransferFnc) const;

};

} /* namespace sparse_net_library */
#endif // Dense_net_weight_initializer_H
