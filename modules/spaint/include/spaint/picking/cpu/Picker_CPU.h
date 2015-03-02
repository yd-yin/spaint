/**
 * spaint: Picker_CPU.h
 */

#ifndef H_SPAINT_PICKER_CPU
#define H_SPAINT_PICKER_CPU

#include "../interface/Picker.h"

namespace spaint {

/**
 * \brief An instance of this class can be used to pick an individual point in the scene using the CPU.
 */
class Picker_CPU : public Picker
{
  //#################### PUBLIC MEMBER FUNCTIONS ####################
public:
  /** Override */
  bool pick(int x, int y, const ITMLib::Objects::ITMRenderState *renderState, ORUtils::MemoryBlock<Vector3f>& pickPointMB) const;
};

}

#endif
