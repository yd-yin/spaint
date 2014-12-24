/**
 * spaint: CompositeCamera.cpp
 */

#include "cameras/CompositeCamera.h"

#include <stdexcept>

namespace spaint {

//#################### CONSTRUCTORS ####################

CompositeCamera::CompositeCamera(const MoveableCamera_Ptr& primaryCamera)
: m_primaryCamera(primaryCamera)
{}

//#################### PUBLIC MEMBER FUNCTIONS ####################

void CompositeCamera::add_secondary_camera(const std::string& name, const Camera_CPtr& camera)
{
  bool result = m_secondaryCameras.insert(std::make_pair(name, camera)).second;
  if(!result) throw std::runtime_error("The composite already contains a camera named '" + name + "'");
}

const Camera_CPtr& CompositeCamera::get_secondary_camera(const std::string& name) const
{
  std::map<std::string,Camera_CPtr>::const_iterator it = m_secondaryCameras.find(name);
  if(it == m_secondaryCameras.end()) throw std::runtime_error("The composite does not contain a camera named '" +  name + "'");
  return it->second;
}

CompositeCamera& CompositeCamera::move_n(float delta)
{
  m_primaryCamera->move_n(delta);
  return *this;
}

CompositeCamera& CompositeCamera::move_u(float delta)
{
  m_primaryCamera->move_u(delta);
  return *this;
}

CompositeCamera& CompositeCamera::move_v(float delta)
{
  m_primaryCamera->move_v(delta);
  return *this;
}

Eigen::Vector3f CompositeCamera::n() const
{
  return m_primaryCamera->n();
}

Eigen::Vector3f CompositeCamera::p() const
{
  return m_primaryCamera->p();
}

void CompositeCamera::remove_secondary_camera(const std::string& name)
{
  std::map<std::string,Camera_CPtr>::iterator it = m_secondaryCameras.find(name);
  if(it == m_secondaryCameras.end()) throw std::runtime_error("The composite does not contain a camera named '" +  name + "'");
  m_secondaryCameras.erase(it);
}

CompositeCamera& CompositeCamera::rotate(const Eigen::Vector3f& axis, float angle)
{
  m_primaryCamera->rotate(axis, angle);
  return *this;
}

Eigen::Vector3f CompositeCamera::u() const
{
  return m_primaryCamera->u();
}

Eigen::Vector3f CompositeCamera::v() const
{
  return m_primaryCamera->v();
}

}
