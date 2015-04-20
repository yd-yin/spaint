/**
 * spaint: ImageProcessor_Shared.h
 */

#ifndef H_SPAINT_IMAGEPROCESSOR_SHARED
#define H_SPAINT_IMAGEPROCESSOR_SHARED

namespace spaint {

//#################### SHARED HELPER FUNCTIONS ####################

/**
 * \brief Shades a pixel with the absolute difference between two other images if the values in the two other images are greater than or equal to zero.
 *
 * \param destination   The location into which to write the computed absolute difference.
 * \param firstInput    The first value.
 * \param secondInput   The second value.
 */
_CPU_AND_GPU_CODE_
inline void shade_pixel_absolute_difference(float *destination, float firstInput, float secondInput) 
{
  if((firstInput < 0) || (secondInput < 0))
  {
    *destination = -1.0f;
  }
  else
  {
    *destination = fabs(firstInput - secondInput);
  }
}

/**
 * \brief Shades a pixel on comparison.
 */
template <typename T>
_CPU_AND_GPU_CODE_
inline void shade_pixel_on_comparison(float *output, float input, float comparator, T comparisonOperator, float value)
{
  switch(comparisonOperator)
  {
    case ImageProcessor::GREATER:
    {
      if(input > comparator) *output = value;
      else *output = input;
      break;
    }
    case ImageProcessor::LESS:
    {
      if(input < comparator) *output = value;
      else *output = input;
      break;
    }
    default:
    {
      // This should never happen.
      //throw std::runtime_error("Unknown comparison type");
      printf("Unknown comparison type");
      break;
    }
  }
}

/*
template <ImageProcessor::ComparisonOperator T>
_CPU_AND_GPU_CODE_
inline void shade_pixel_on_comparison(float *output, float input, float comparator, float value)
{}

template<>
_CPU_AND_GPU_CODE_
inline void shade_pixel_on_comparison<ImageProcessor::GREATER>(float *output, float input, float comparator, float value)
{
  if(input > comparator) *output = value;
  else *output = input;
}
*/


/**
 * \brief Shades invalid pixels (one with a value of less than zero) with a specified value.
 */
_CPU_AND_GPU_CODE_
inline void shade_invalid_pixels(float *destination, float value)
{
  if(*destination < 0) *destination = value;
}

}

#endif
