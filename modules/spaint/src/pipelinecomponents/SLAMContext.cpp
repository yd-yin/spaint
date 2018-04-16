/**
 * spaint: SLAMContext.cpp
 * Copyright (c) Torr Vision Group, University of Oxford, 2016. All rights reserved.
 */

#include "pipelinecomponents/SLAMContext.h"

#include <tvgutil/containers/MapUtil.h>
using namespace tvgutil;

namespace spaint {

//#################### PUBLIC MEMBER FUNCTIONS ####################

itmx::RefiningRelocaliser_Ptr& SLAMContext::get_fast_relocaliser(const std::string& sceneID)
{
  return m_relocalisers[sceneID + "_Fast"];
}

itmx::RefiningRelocaliser_CPtr SLAMContext::get_fast_relocaliser(const std::string& sceneID) const
{
  return MapUtil::lookup(m_relocalisers, sceneID + "_Fast");
}

itmx::RefiningRelocaliser_Ptr& SLAMContext::get_intermediate_relocaliser(const std::string& sceneID)
{
  return m_relocalisers[sceneID + "_Intermediate"];
}

itmx::RefiningRelocaliser_CPtr SLAMContext::get_intermediate_relocaliser(const std::string& sceneID) const
{
  return MapUtil::lookup(m_relocalisers, sceneID + "_Intermediate");
}

itmx::RefiningRelocaliser_Ptr& SLAMContext::get_relocaliser(const std::string& sceneID)
{
  return m_relocalisers[sceneID];
}

itmx::RefiningRelocaliser_CPtr SLAMContext::get_relocaliser(const std::string& sceneID) const
{
  return MapUtil::lookup(m_relocalisers, sceneID);
}

const SLAMState_Ptr& SLAMContext::get_slam_state(const std::string& sceneID)
{
  SLAMState_Ptr& result = m_slamStates[sceneID];
  if(!result) result.reset(new SLAMState);
  return result;
}

SLAMState_CPtr SLAMContext::get_slam_state(const std::string& sceneID) const
{
  return MapUtil::lookup(m_slamStates, sceneID);
}

}
