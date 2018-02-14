/**
 * spaintgui: MultiScenePipeline.cpp
 * Copyright (c) Torr Vision Group, University of Oxford, 2016. All rights reserved.
 */

#include "MultiScenePipeline.h"
using namespace InputSource;
using namespace ITMLib;
using namespace itmx;
using namespace spaint;

#include <boost/bind.hpp>
#include <boost/filesystem.hpp>
namespace bf = boost::filesystem;

#include <tvgutil/containers/MapUtil.h>
using namespace tvgutil;

//#################### CONSTRUCTORS ####################

MultiScenePipeline::MultiScenePipeline(const std::string& type, const Settings_Ptr& settings, const std::string& resourcesDir,
                                       size_t maxLabelCount, const MappingServer_Ptr& mappingServer)
: m_mode(MODE_NORMAL), m_type(type)
{
  // Make sure that we're not trying to run on the GPU if CUDA support isn't enabled.
#ifndef WITH_CUDA
  if(settings->deviceType == ITMLibSettings::DEVICE_CUDA)
  {
    std::cerr << "[spaint] CUDA support unavailable, reverting to the CPU implementation of InfiniTAM\n";
    settings->deviceType = ITMLibSettings::DEVICE_CPU;
  }
#endif

  // Set up the spaint model.
  m_model.reset(new Model(settings, resourcesDir, maxLabelCount, mappingServer));
}

//#################### DESTRUCTOR ####################

MultiScenePipeline::~MultiScenePipeline() {}

//#################### PUBLIC MEMBER FUNCTIONS ####################

bool MultiScenePipeline::get_fusion_enabled(const std::string& sceneID) const
{
  return MapUtil::lookup(m_slamComponents, sceneID)->get_fusion_enabled();
}

MultiScenePipeline::Mode MultiScenePipeline::get_mode() const
{
  return m_mode;
}

const Model_Ptr& MultiScenePipeline::get_model()
{
  return m_model;
}

Model_CPtr MultiScenePipeline::get_model() const
{
  return m_model;
}

const std::string& MultiScenePipeline::get_type() const
{
  return m_type;
}

void MultiScenePipeline::reset_forest(const std::string& sceneID)
{
  MapUtil::call_if_found(m_semanticSegmentationComponents, sceneID, boost::bind(&SemanticSegmentationComponent::reset_forest, _1));
}

void MultiScenePipeline::reset_scene(const std::string& sceneID)
{
  MapUtil::call_if_found(m_slamComponents, sceneID, boost::bind(&SLAMComponent::reset_scene, _1));
}

size_t MultiScenePipeline::run_main_section()
{
  size_t result = 0;
  for(std::map<std::string,SLAMComponent_Ptr>::const_iterator it = m_slamComponents.begin(), iend = m_slamComponents.end(); it != iend; ++it)
  {
    if(it->second->process_frame()) ++result;
  }
  return result;
}

void MultiScenePipeline::run_mode_specific_section(const std::string& sceneID, const VoxelRenderState_CPtr& renderState)
{
  switch(m_mode)
  {
    case MODE_FEATURE_INSPECTION:
      MapUtil::call_if_found(m_semanticSegmentationComponents, sceneID, boost::bind(&SemanticSegmentationComponent::run_feature_inspection, _1, renderState));
      break;
    case MODE_PREDICTION:
      MapUtil::call_if_found(m_semanticSegmentationComponents, sceneID, boost::bind(&SemanticSegmentationComponent::run_prediction, _1, renderState));
      break;
    case MODE_PROPAGATION:
      MapUtil::call_if_found(m_propagationComponents, sceneID, boost::bind(&PropagationComponent::run, _1, renderState));
      break;
    case MODE_SEGMENTATION:
      MapUtil::call_if_found(m_objectSegmentationComponents, sceneID, boost::bind(&ObjectSegmentationComponent::run_segmentation, _1, renderState));
      break;
    case MODE_SEGMENTATION_TRAINING:
      MapUtil::call_if_found(m_objectSegmentationComponents, sceneID, boost::bind(&ObjectSegmentationComponent::run_segmentation_training, _1, renderState));
      break;
    case MODE_SMOOTHING:
      MapUtil::call_if_found(m_smoothingComponents, sceneID, boost::bind(&SmoothingComponent::run, _1, renderState));
      break;
    case MODE_TRAIN_AND_PREDICT:
    {
      static bool trainThisFrame = false;
      trainThisFrame = !trainThisFrame;

      if(trainThisFrame) MapUtil::call_if_found(m_semanticSegmentationComponents, sceneID, boost::bind(&SemanticSegmentationComponent::run_training, _1, renderState));
      else MapUtil::call_if_found(m_semanticSegmentationComponents, sceneID, boost::bind(&SemanticSegmentationComponent::run_prediction, _1, renderState));

      break;
    }
    case MODE_TRAINING:
      MapUtil::call_if_found(m_semanticSegmentationComponents, sceneID, boost::bind(&SemanticSegmentationComponent::run_training, _1, renderState));
      break;
    default:
      break;
  }
}

void MultiScenePipeline::save_models(const bf::path& outputDir) const
{
  // Make sure that the output directory exists.
  bf::create_directories(outputDir);

  // Save the models for each scene into a separate subdirectory.
  for(std::map<std::string,SLAMComponent_Ptr>::const_iterator it = m_slamComponents.begin(), iend = m_slamComponents.end(); it != iend; ++it)
  {
    const bf::path scenePath = outputDir / it->first;
    std::cout << "Saving models for " << it->first << " in: " << scenePath << '\n';
    it->second->save_models(scenePath.string());
  }
}

void MultiScenePipeline::set_detect_fiducials(const std::string& sceneID, bool detectFiducials)
{
  MapUtil::lookup(m_slamComponents, sceneID)->set_detect_fiducials(detectFiducials);
}

void MultiScenePipeline::set_fusion_enabled(const std::string& sceneID, bool fusionEnabled)
{
  MapUtil::lookup(m_slamComponents, sceneID)->set_fusion_enabled(fusionEnabled);
}

void MultiScenePipeline::set_mapping_client(const std::string& sceneID, const itmx::MappingClient_Ptr& mappingClient)
{
  MapUtil::call_if_found(m_slamComponents, sceneID, boost::bind(&SLAMComponent::set_mapping_client, _1, mappingClient));
}

void MultiScenePipeline::toggle_segmentation_output()
{
  MapUtil::call_if_found(m_objectSegmentationComponents, Model::get_world_scene_id(), boost::bind(&ObjectSegmentationComponent::toggle_output, _1));
}

void MultiScenePipeline::update_raycast_result_size(int raycastResultSize)
{
  for(std::map<std::string,PropagationComponent_Ptr>::const_iterator it = m_propagationComponents.begin(), iend = m_propagationComponents.end(); it != iend; ++it)
  {
    it->second->reset_label_propagator(raycastResultSize);
  }

  for(std::map<std::string,SemanticSegmentationComponent_Ptr>::const_iterator it = m_semanticSegmentationComponents.begin(), iend = m_semanticSegmentationComponents.end(); it != iend; ++it)
  {
    it->second->reset_voxel_samplers(raycastResultSize);
  }
}

//#################### PROTECTED MEMBER FUNCTIONS ####################

void MultiScenePipeline::load_models(const SLAMComponent_Ptr& slamComponent, const std::string& inputDir)
{
  std::cout << "Loading models for " << slamComponent->get_scene_id() << " from: " << inputDir << std::endl;
  slamComponent->load_models(inputDir);
}
