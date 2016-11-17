/**
 * spaint: GPUReservoir_CUDA.h
 * Copyright (c) Torr Vision Group, University of Oxford, 2016. All rights reserved.
 */

#ifndef H_SPAINT_GPURESERVOIRCUDA
#define H_SPAINT_GPURESERVOIRCUDA

#include "../interface/GPUReservoir.h"

#include <curand_kernel.h>

namespace spaint
{
class GPUReservoir_CUDA: public GPUReservoir
{
public:
  typedef curandState_t RandomState;
  typedef ORUtils::MemoryBlock<RandomState> RandomStateMemoryBlock;
  typedef boost::shared_ptr<RandomStateMemoryBlock> RandomStateMemoryBlock_Ptr;
  typedef boost::shared_ptr<const RandomStateMemoryBlock> RandomStateMemoryBlock_CPtr;

  GPUReservoir_CUDA(size_t capacity, size_t nbLeaves, uint32_t rngSeed = 42);

  virtual void add_examples(const RGBDPatchFeatureImage_CPtr &features,
      const LeafIndicesImage_CPtr &leafIndices);
  virtual void clear();

private:
  RandomStateMemoryBlock_Ptr m_randomStates; // Maybe in the cuda class

  void init_random();
};
}

#endif
