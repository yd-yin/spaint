/**
 * spaintgui: SLAMPipeline.cpp
 * Copyright (c) Torr Vision Group, University of Oxford, 2016. All rights reserved.
 */

#include "SLAMPipeline.h"
using namespace spaint;

//#################### CONSTRUCTORS ####################

SLAMPipeline::SLAMPipeline(const Settings_Ptr& settings, const std::string& resourcesDir,
                           const CompositeImageSourceEngine_Ptr& imageSourceEngine, const std::string& trackerConfig,
                           SLAMComponent::MappingMode mappingMode, SLAMComponent::TrackingMode trackingMode,
                           const FiducialDetector_CPtr& fiducialDetector, bool detectFiducials)
  // Note: A minimum of 2 labels is required (background and foreground).
: MultiScenePipeline("slam", settings, resourcesDir, 2)
{
  const std::string sceneID = Model::get_world_scene_id();
  m_slamComponents[sceneID].reset(new SLAMComponent(m_model, sceneID, imageSourceEngine, trackerConfig, mappingMode, trackingMode, fiducialDetector, detectFiducials));
}

//#################### PUBLIC MEMBER FUNCTIONS ####################

void SLAMPipeline::set_mode(Mode mode)
{
  // The only supported mode.
  m_mode = MODE_NORMAL;
}
