/**
 * spaint: DepthCalculator_CUDA.h
 */

#ifndef H_SPAINT_DEPTHVISUALISER_CUDA
#define H_SPAINT_DEPTHVISUALISER_CUDA

#include "../interface/DepthCalculator.h"

namespace spaint {

/**
 * \brief An instance of this class can be used to render a depth visualisation of an InfiniTAM scene using CUDA.
 */
class DepthCalculator_CUDA : public DepthCalculator
{
  //#################### PUBLIC MEMBER FUNCTIONS #################### 
public:
  /** Override */
  virtual void render_euclidean_distance(ITMFloatImage *outputImage, const ITMLib::Objects::ITMRenderState *renderState, const rigging::SimpleCamera *camera, float voxelSize) const;
  virtual void render_orthographic_distance(ITMFloatImage *outputImage, const ITMLib::Objects::ITMRenderState *renderState, const rigging::SimpleCamera *camera, float voxelSize) const;
};

}

#endif
