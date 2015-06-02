/**
 * spaint: OpenCVUtil.h
 */

#ifndef H_SPAINT_OPENCVUTIL
#define H_SPAINT_OPENCVUTIL

#include <stdexcept>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <ITMLib/Utils/ITMLibDefines.h>

namespace spaint {

/**
 * \brief This class provides helper functions to visualise InfiniTAM and ArrayFire images using OpenCV.
 */
class OpenCVUtil
{
  //#################### PUBLIC ENUMERATIONS ####################
public:
  /**
   * \brief An enumeration containing two possible ways of arranging multidimensional arrays in a single linear array.
   */
  enum Order
  {
    ROW_MAJOR,
    COL_MAJOR
  };

  //#################### PUBLIC STATIC MEMBER FUNCTIONS ####################
public:
  /*
   * \brief Displays an image and scales the pixel values by a specified scaling factor.
   *
   * \param infiniTAMImage  The InfiniTAM image.
   * \param scaleFactor     The factor by which to scale the image pixels.
   * \param windowName      The name of the window in which to display the resulting image.
   */
  static void display_image_and_scale(ITMFloatImage *infiniTAMImage, float scaleFactor, const std::string& windowName);

  /*
   * \brief Displays an image and scales the pixel values to occupy the entire range [0-255].
   *
   * \param infiniTAMImage  The InfiniTAM image.
   * \param windowName      The name of the window in which to display the resulting image.
   */
  static void display_image_scale_to_range(ITMFloatImage *infiniTAMImage, const std::string& windowName);

  /**
   * \brief Displays an image in a window from an array of pixel values.
   *
   * \param windowName   The name of the window.
   * \param pixels       A pointer to the first pixel element in the image.
   * \param width        The width of the image.
   * \param height       The hwight of the image.
   * \param order        Whether the pixel values are arrange din column-major or row-major order.
   */
  static void ocvfig(const std::string& windowName, unsigned char *pixels, int width, int height, Order order);

  //#################### PRIVATE STATIC MEMBER FUNCTIONS ####################
private:
  /**
   * \brief Calculates the minimum and maximum values in an InfiniTAM image.
   *
   * \param itmImageDataPtr The InfiniTAM image data pointer.
   * \param width           The width of the image.
   * \param height          The height of the image.
   * \return                A pair of values indicating the minimum and maximum values found.
   */
  static std::pair<float, float> itm_mat_32SC1_min_max_calculator(float *itmImageDataPtr, int width, int height);
};

}

#endif
