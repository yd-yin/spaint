/**
 * spaint: SimpleCamera.h
 */

#ifndef H_SPAINT_SIMPLECAMERA
#define H_SPAINT_SIMPLECAMERA

#include "Camera.h"

namespace spaint {

/**
 * \brief An instance of this class represents a simple camera in 3D space.
 */
class SimpleCamera : public Camera
{
  //#################### PRIVATE VARIABLES ####################
private:
  /** A vector pointing in the direction faced by the camera. */
  Eigen::Vector3f m_n;

  /** The position of the camera. */
  Eigen::Vector3f m_position;

  /** A vector pointing to the left of the camera. */
  Eigen::Vector3f m_u;

  /** A vector pointing to the top of the camera. */
  Eigen::Vector3f m_v;

  //#################### CONSTRUCTORS ####################
public:
  /**
   * \brief Constructs a simple camera.
   *
   * \param position  The position of the camera.
   * \param look      A vector pointing in the direction faced by the camera.
   * \param up        The "up" direction for the camera.
   */
  SimpleCamera(const Eigen::Vector3f& position, const Eigen::Vector3f& look, const Eigen::Vector3f& up);

  //#################### PUBLIC MEMBER FUNCTIONS ####################
public:
  /** Override */
  SimpleCamera& move_n(float delta);

  /** Override */
  SimpleCamera& move_u(float delta);

  /** Override */
  SimpleCamera& move_v(float delta);

  /** Override */
  Eigen::Vector3f n() const;

  /** Override */
  Eigen::Vector3f p() const;

  /** Override */
  SimpleCamera& rotate(const Eigen::Vector3f& axis, float angle);

  /** Override */
  Eigen::Vector3f u() const;

  /** Override */
  Eigen::Vector3f v() const;
};

}

#endif
