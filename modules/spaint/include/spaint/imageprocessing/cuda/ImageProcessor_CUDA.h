/**
 * spaint: ImageProcessor_CUDA.h
 */

#ifndef H_SPAINT_IMAGEPROCESSOR_CUDA
#define H_SPAINT_IMAGEPROCESSOR_CUDA

#include "../interface/ImageProcessor.h"

namespace spaint {

/**
 * \brief An instance of this class may be used to apply image processing algorithms to images with CUDA.
 */
class ImageProcessor_CUDA : public ImageProcessor
{
  //#################### PUBLIC MEMBER FUNCTIONS ####################
public:
  /** Override. */
  virtual void absolute_difference_calculator(ITMFloatImage *outputImage, ITMFloatImage *firstInputImage, ITMFloatImage *secondInputImage) const;

  /** Override. */
  virtual void absolute_difference_calculator(af::array *outputImage, ITMFloatImage *firstInputImage, ITMFloatImage *secondInputImage) const;

  /** Override. */
  virtual void pixel_setter(ITMFloatImage *output, ITMFloatImage *input, float comparator, ComparisonOperator comparisonOperator, float value) const;

  /** Override. */
  virtual void set_invalid_regions(ITMFloatImage *image, float value) const;
};

}

#endif
