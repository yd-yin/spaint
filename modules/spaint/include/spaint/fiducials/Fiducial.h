/**
 * spaint: Fiducial.h
 * Copyright (c) Torr Vision Group, University of Oxford, 2016. All rights reserved.
 */

#ifndef H_SPAINT_FIDUCIAL
#define H_SPAINT_FIDUCIAL

#include <boost/shared_ptr.hpp>

#include "FiducialMeasurement.h"

namespace spaint {

/**
 * \brief An instance of this class represents a fiducial (a reference marker in a 3D scene).
 */
class Fiducial
{
  //#################### PRIVATE VARIABLES ####################
private:
  /** The ID of the fiducial. */
  std::string m_id;

  /** The pose of the fiducial in world space. */
  ORUtils::SE3Pose m_pose;

  //#################### CONSTRUCTORS ####################
public:
  /**
   * \brief Constructs a fiducial.
   *
   * \param id    The ID of the fiducial.
   * \param pose  The pose of the fiducial in world space.
   */
  Fiducial(const std::string& id, const ORUtils::SE3Pose& pose);

  //#################### PUBLIC MEMBER FUNCTIONS ####################
public:
  /**
   * \brief Gets the ID of the fiducial.
   *
   * \return  The ID of the fiducial.
   */
  const std::string& id() const;

  /**
   * \brief Updates the fiducial based on information from a new measurement.
   *
   * \param measurement         The new measurement.
   * \throws std::runtime_error If the fiducial and the measurement do not have the same ID,
   *                            or if the measurement does not contain a valid world pose.
   */
  void integrate(const FiducialMeasurement& measurement);

  /**
   * \brief Gets the pose of the fiducial in world space.
   *
   * \return  The pose of the fiducial in world space.
   */
  const ORUtils::SE3Pose& pose() const;
};

}

#endif
