/**
 * spaint: FiducialDetector.h
 * Copyright (c) Torr Vision Group, University of Oxford, 2016. All rights reserved.
 */

#ifndef H_SPAINT_FIDUCIALDETECTOR
#define H_SPAINT_FIDUCIALDETECTOR

#include <map>

#include <itmx/base/ITMObjectPtrTypes.h>

#include "FiducialMeasurement.h"

namespace spaint {

/**
 * \brief An instance of a class deriving from this one can be used to detect fiducials in a 3D scene.
 */
class FiducialDetector
{
  //#################### ENUMERATIONS ####################
public:
  /**
   * \brief The values of this enumeration denote the different fiducial pose estimation modes that are supported.
   */
  enum PoseEstimationMode
  {
    /** Estimate the poses of the fiducials from the live colour image. */
    PEM_COLOUR,

    /** Estimate the poses of the fiducials from the live depth image. */
    PEM_DEPTH,

    /** Estimate the poses of the fiducials from a depth raycast of the scene. */
    PEM_RAYCAST
  };

  //#################### DESTRUCTOR ####################
public:
  /**
   * \brief Destroys the fiducial detector.
   */
  virtual ~FiducialDetector() {}

  //#################### PUBLIC ABSTRACT MEMBER FUNCTIONS ####################
public:
  /**
   * \brief Detects fiducials in a view of a 3D scene.
   *
   * \param view                A view of the 3D scene.
   * \param pose                The pose from which the view was captured.
   * \param renderState         A render state corresponding to the camera pose.
   * \param poseEstimationMode  The mode to use when estimating the poses of the fiducials.
   * \return                    Measurements of the fiducials (if any) that have been detected in the view.
   */
  virtual std::map<std::string,FiducialMeasurement> detect_fiducials(const View_CPtr& view, const ORUtils::SE3Pose& pose, const VoxelRenderState_CPtr& renderState,
                                                                     PoseEstimationMode poseEstimationMode) const = 0;

  //#################### PROTECTED STATIC MEMBER FUNCTIONS ####################
protected:
  /**
   * \brief Attempts to make a pose matrix from three corner points.
   *
   * \param v0  The first corner point.
   * \param v1  The second corner point.
   * \param v2  The third corner point.
   * \return    The pose matrix, if all three corner points exist, or boost::none otherwise.
   */
  static boost::optional<ORUtils::SE3Pose> make_pose_from_corners(const boost::optional<Vector3f>& v0,
                                                                  const boost::optional<Vector3f>& v1,
                                                                  const boost::optional<Vector3f>& v2);
};

//#################### TYPEDEFS ####################

typedef boost::shared_ptr<const FiducialDetector> FiducialDetector_CPtr;

}

#endif
