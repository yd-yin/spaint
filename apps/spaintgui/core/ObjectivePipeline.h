/**
 * spaintgui: ObjectivePipeline.h
 * Copyright (c) Torr Vision Group, University of Oxford, 2016. All rights reserved.
 */

#ifndef H_SPAINTGUI_OBJECTIVEPIPELINE
#define H_SPAINTGUI_OBJECTIVEPIPELINE

#include "MultiScenePipeline.h"

/**
 * \brief An instance of this class represents an ObjectivePaint processing pipeline.
 */
class ObjectivePipeline : public MultiScenePipeline
{
  //#################### CONSTRUCTORS ####################
public:
  /**
   * \brief Constructs an objective pipeline.
   *
   * \param settings          The settings to use for InfiniTAM.
   * \param resourcesDir      The path to the resources directory.
   * \param maxLabelCount     The maximum number of labels that can be in use.
   * \param imageSourceEngine The engine used to provide input images to the SLAM component for the world scene.
   * \param trackerConfig     The tracker configuration to use when reconstructing either scene.
   * \param mappingMode       The mapping mode that the world scene's SLAM component should use.
   * \param trackingMode      The tracking mode that the world scene's SLAM component should use.
   * \param detectFiducials   Whether or not to initially detect fiducials in the 3D scene.
   * \param mirrorWorldPose   Whether or not to mirror the world pose when reconstructing the object.
   */
  ObjectivePipeline(const Settings_Ptr& settings, const std::string& resourcesDir, size_t maxLabelCount,
                    const CompositeImageSourceEngine_Ptr& imageSourceEngine, const std::string& trackerConfig,
                    spaint::SLAMComponent::MappingMode mappingMode = spaint::SLAMComponent::MAP_VOXELS_ONLY,
                    spaint::SLAMComponent::TrackingMode trackingMode = spaint::SLAMComponent::TRACK_VOXELS,
                    bool detectFiducials = false, bool mirrorWorldPose = true);

  //#################### PUBLIC MEMBER FUNCTIONS ####################
public:
  /** Override */
  virtual void set_mode(Mode mode);
};

#endif
