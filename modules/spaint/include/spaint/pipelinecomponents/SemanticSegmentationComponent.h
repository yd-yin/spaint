/**
 * spaint: SemanticSegmentationComponent.h
 * Copyright (c) Torr Vision Group, University of Oxford, 2016. All rights reserved.
 */

#ifndef H_SPAINT_SEMANTICSEGMENTATIONCOMPONENT
#define H_SPAINT_SEMANTICSEGMENTATIONCOMPONENT

#include <ITMLib/Objects/RenderStates/ITMRenderState.h>

#include <rafl/core/RandomForest.h>

#include "SemanticSegmentationContext.h"
#include "../features/interface/FeatureCalculator.h"
#include "../sampling/interface/PerLabelVoxelSampler.h"
#include "../sampling/interface/UniformVoxelSampler.h"
#include "../selectors/Selector.h"

namespace spaint {

/**
 * \brief TODO
 */
class SemanticSegmentationComponent
{
  //#################### TYPEDEFS ####################
private:
  typedef boost::shared_ptr<rafl::RandomForest<SpaintVoxel::Label> > RandomForest_Ptr;
  typedef boost::shared_ptr<const ITMLib::ITMRenderState> RenderState_CPtr;
  typedef boost::shared_ptr<const ITMLib::ITMLibSettings> Settings_CPtr;

  //#################### PRIVATE VARIABLES ####################
private:
  /** TODO */
  SemanticSegmentationContext_Ptr m_context;

  /** The feature calculator. */
  FeatureCalculator_CPtr m_featureCalculator;

  /** The random forest. */
  RandomForest_Ptr m_forest;

  /** The maximum number of labels that can be in use. */
  size_t m_maxLabelCount;

  /** The maximum number of voxels for which to predict labels each frame. */
  size_t m_maxPredictionVoxelCount;

  /** The maximum number of voxels per label from which to train each frame. */
  size_t m_maxTrainingVoxelsPerLabel;

  /** The side length of a VOP patch (must be odd). */
  size_t m_patchSize;

  /** A memory block in which to store the feature vectors computed for the various voxels during prediction. */
  boost::shared_ptr<ORUtils::MemoryBlock<float> > m_predictionFeaturesMB;

  /** A memory block in which to store the labels predicted for the various voxels. */
  boost::shared_ptr<ORUtils::MemoryBlock<SpaintVoxel::PackedLabel> > m_predictionLabelsMB;

  /** The voxel sampler used in prediction mode. */
  UniformVoxelSampler_CPtr m_predictionSampler;

  /** A memory block in which to store the locations of the voxels sampled for prediction purposes. */
  Selector::Selection_Ptr m_predictionVoxelLocationsMB;

  /** The path to the resources directory. */
  std::string m_resourcesDir;

  /** A memory block in which to store the feature vectors computed for the various voxels during training. */
  boost::shared_ptr<ORUtils::MemoryBlock<float> > m_trainingFeaturesMB;

  /** A memory block in which to store a mask indicating which labels are currently in use and from which we want to train. */
  boost::shared_ptr<ORUtils::MemoryBlock<bool> > m_trainingLabelMaskMB;

  /** The voxel sampler used in training mode. */
  PerLabelVoxelSampler_CPtr m_trainingSampler;

  /** A memory block in which to store the number of voxels sampled for each label for training purposes. */
  boost::shared_ptr<ORUtils::MemoryBlock<unsigned int> > m_trainingVoxelCountsMB;

  /** A memory block in which to store the locations of the voxels sampled for training purposes. */
  Selector::Selection_Ptr m_trainingVoxelLocationsMB;

  //#################### CONSTRUCTORS ####################
public:
  /**
   * \brief TODO
   */
  SemanticSegmentationComponent(const SemanticSegmentationContext_Ptr& context, const Vector2i& depthImageSize, unsigned int seed,
                                const Settings_CPtr& settings, const std::string& resourcesDir, size_t maxLabelCount);

  //#################### PUBLIC MEMBER FUNCTIONS ####################
public:
  /**
   * \brief Resets the random forest.
   */
  void reset_forest();

  /** TODO */
  void run_feature_inspection(const RenderState_CPtr& renderState);

  /** TODO */
  void run_prediction(const RenderState_CPtr& samplingRenderState);

  /** TODO */
  void run_training(const RenderState_CPtr& samplingRenderState);
};

//#################### TYPEDEFS ####################

typedef boost::shared_ptr<SemanticSegmentationComponent> SemanticSegmentationComponent_Ptr;

}

#endif
