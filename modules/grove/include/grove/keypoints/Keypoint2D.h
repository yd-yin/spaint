/**
 * grove: Keypoint2D.h
 * Copyright (c) Torr Vision Group, University of Oxford, 2017. All rights reserved.
 */

#ifndef H_GROVE_KEYPOINT2D
#define H_GROVE_KEYPOINT2D

#include <boost/shared_ptr.hpp>

#include <ORUtils/Image.h>
#include <ORUtils/Math.h>

namespace grove {

/**
 * \brief An instance of this struct represents a 2D keypoint.
 */
struct Keypoint2D
{
  //#################### PUBLIC VARIABLES ####################

  /** The keypoint's position in the image. */
  Vector2f position;

  /** A flag indicating whether or not the keypoint is valid. */
  bool valid;
};

//#################### TYPEDEFS ####################

typedef ORUtils::Image<Keypoint2D> Keypoint2DImage;
typedef boost::shared_ptr<Keypoint2DImage> Keypoint2DImage_Ptr;
typedef boost::shared_ptr<const Keypoint2DImage> Keypoint2DImage_CPtr;

}

#endif
