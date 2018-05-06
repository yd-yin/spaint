/**
 * grove: ScoreRelocaliser.cpp
 * Copyright (c) Torr Vision Group, University of Oxford, 2017. All rights reserved.
 */

#include "relocalisation/interface/ScoreRelocaliser.h"

#include <boost/filesystem.hpp>
namespace bf = boost::filesystem;

#include <ITMLib/Engines/LowLevel/ITMLowLevelEngineFactory.h>
using namespace ITMLib;

#include <itmx/base/MemoryBlockFactory.h>
using namespace itmx;

#include <tvgutil/misc/SettingsContainer.h>
using namespace tvgutil;

#include "clustering/ExampleClustererFactory.h"
#include "features/FeatureCalculatorFactory.h"
#include "forests/DecisionForestFactory.h"
#include "ransac/PreemptiveRansacFactory.h"
#include "reservoirs/ExampleReservoirsFactory.h"

namespace grove {

//#################### CONSTRUCTORS ####################

ScoreRelocaliser::ScoreRelocaliser(const std::string& forestFilename, const SettingsContainer_CPtr& settings, DeviceType deviceType)
: m_deviceType(deviceType), m_settings(settings)
{
  const std::string settingsNamespace = "ScoreRelocaliser.";

  // Determine the top-level parameters for the relocaliser.
  m_maxRelocalisationsToOutput = m_settings->get_first_value<uint32_t>(settingsNamespace + "maxRelocalisationsToOutput", 1);

  // Determine the reservoir-related parameters.
  m_maxReservoirsToUpdate = m_settings->get_first_value<uint32_t>(settingsNamespace + "maxReservoirsToUpdate", 256);  // Update the modes associated with this number of reservoirs for each train/update call.
  m_reservoirCapacity = m_settings->get_first_value<uint32_t>(settingsNamespace + "reservoirCapacity", 1024);
  m_rngSeed = m_settings->get_first_value<uint32_t>(settingsNamespace + "rngSeed", 42);

  // Determine the clustering-related parameters (the defaults are tentative values that seem to work).
  m_clustererSigma = m_settings->get_first_value<float>(settingsNamespace + "clustererSigma", 0.1f);
  m_clustererTau = m_settings->get_first_value<float>(settingsNamespace + "clustererTau", 0.05f);
  m_maxClusterCount = m_settings->get_first_value<uint32_t>(settingsNamespace + "maxClusterCount", ScorePrediction::Capacity);
  m_minClusterSize = m_settings->get_first_value<uint32_t>(settingsNamespace + "minClusterSize", 20);

  // Check that the maximum number of clusters to store in each leaf is within range.
  if(m_maxClusterCount > ScorePrediction::Capacity)
  {
    throw std::invalid_argument(settingsNamespace + "maxClusterCount > ScorePrediction::Capacity");
  }

  // Allocate the internal images.
  MemoryBlockFactory& mbf = MemoryBlockFactory::instance();
  m_descriptorsImage = mbf.make_image<DescriptorType>();
  m_keypointsImage = mbf.make_image<ExampleType>();
  m_leafIndicesImage = mbf.make_image<LeafIndices>();
  m_predictionsImage = mbf.make_image<ScorePrediction>();

  // Instantiate the sub-algorithms.
  m_featureCalculator = FeatureCalculatorFactory::make_da_rgbd_patch_feature_calculator(deviceType);
  m_lowLevelEngine.reset(ITMLowLevelEngineFactory::MakeLowLevelEngine(deviceType));
  m_scoreForest = DecisionForestFactory<DescriptorType,FOREST_TREE_COUNT>::make_forest(forestFilename, deviceType);
  m_reservoirCount = m_scoreForest->get_nb_leaves();
  m_exampleClusterer = ExampleClustererFactory<ExampleType,ClusterType,PredictionType::Capacity>::make_clusterer(
    m_clustererSigma, m_clustererTau, m_maxClusterCount, m_minClusterSize, deviceType
  );
  m_preemptiveRansac = PreemptiveRansacFactory::make_preemptive_ransac(settings, deviceType);

  // Set up the relocaliser's internal state.
  m_relocaliserState.reset(new ScoreRelocaliserState);
  reset();
}

//#################### DESTRUCTOR ####################

ScoreRelocaliser::~ScoreRelocaliser() {}

//#################### PUBLIC MEMBER FUNCTIONS ####################

void ScoreRelocaliser::finish_training()
{
  // First update all of the clusters.
  update_all_clusters();

  // Then kill the contents of the reservoirs (we won't need them any more).
  m_relocaliserState->exampleReservoirs.reset();
  m_relocaliserState->lastFeaturesAddedStartIdx = 0;
  m_relocaliserState->reservoirUpdateStartIdx = 0;
}

void ScoreRelocaliser::get_best_poses(std::vector<PoseCandidate>& poseCandidates) const
{
  m_preemptiveRansac->get_best_poses(poseCandidates);
}

Keypoint3DColourImage_CPtr ScoreRelocaliser::get_keypoints_image() const
{
  return m_keypointsImage;
}

ScorePrediction ScoreRelocaliser::get_prediction(uint32_t treeIdx, uint32_t leafIdx) const
{
  // Ensure that the specified leaf is valid (throw if not).
  ensure_valid_leaf(treeIdx, leafIdx);

  // Look up the prediction associated with the leaf and return it.
  const MemoryDeviceType memoryType = m_deviceType == DEVICE_CUDA ? MEMORYDEVICE_CUDA : MEMORYDEVICE_CPU;
  return m_relocaliserState->predictionsBlock->GetElement(leafIdx * m_scoreForest->get_nb_trees() + treeIdx, memoryType);
}

ScorePredictionsImage_CPtr ScoreRelocaliser::get_predictions_image() const
{
  return m_predictionsImage;
}

ScoreRelocaliserState_Ptr ScoreRelocaliser::get_relocaliser_state()
{
  return m_relocaliserState;
}

ScoreRelocaliserState_CPtr ScoreRelocaliser::get_relocaliser_state() const
{
  return m_relocaliserState;
}

std::vector<Keypoint3DColour> ScoreRelocaliser::get_reservoir_contents(uint32_t treeIdx, uint32_t leafIdx) const
{
  // Ensure that the specified leaf is valid (throw if not).
  ensure_valid_leaf(treeIdx, leafIdx);

  // Look up the size of the reservoir associated with the leaf.
  const MemoryDeviceType memoryType = m_deviceType == DEVICE_CUDA ? MEMORYDEVICE_CUDA : MEMORYDEVICE_CPU;
  const uint32_t linearReservoirIdx = leafIdx * m_scoreForest->get_nb_trees() + treeIdx;
  const uint32_t reservoirSize = m_relocaliserState->exampleReservoirs->get_reservoir_sizes()->GetElement(linearReservoirIdx, memoryType);

  // Copy the contents of the reservoir into a suitably-sized buffer and return it.
  if(m_deviceType == DEVICE_CUDA) m_relocaliserState->exampleReservoirs->get_reservoirs()->UpdateHostFromDevice();
  const Keypoint3DColour *reservoirsData = m_relocaliserState->exampleReservoirs->get_reservoirs()->GetData(MEMORYDEVICE_CPU);
  const uint32_t reservoirCapacity = m_relocaliserState->exampleReservoirs->get_reservoir_capacity();

  std::vector<Keypoint3DColour> reservoirContents;
  reservoirContents.reserve(reservoirSize);

  for(uint32_t i = 0; i < reservoirSize; ++i)
  {
    reservoirContents.push_back(reservoirsData[linearReservoirIdx * reservoirCapacity + i]);
  }

  return reservoirContents;
}

void ScoreRelocaliser::load_from_disk(const std::string& inputFolder)
{
  // Load the relocaliser's internal state from disk.
  m_relocaliserState->load_from_disk(inputFolder);
}

std::vector<Relocaliser::Result> ScoreRelocaliser::relocalise(const ITMUChar4Image *colourImage, const ITMFloatImage *depthImage, const Vector4f& depthIntrinsics) const
{
  std::vector<Result> results;

  // Try to estimate a pose only if we have enough valid depth values.
  if(m_lowLevelEngine->CountValidDepths(depthImage) > m_preemptiveRansac->get_min_nb_required_points())
  {
    // First: select keypoints and compute descriptors.
    m_featureCalculator->compute_keypoints_and_features(colourImage, depthImage, depthIntrinsics, m_keypointsImage.get(), m_descriptorsImage.get());

    // Second: find all the leaves associated to the keypoints.
    m_scoreForest->find_leaves(m_descriptorsImage, m_leafIndicesImage);

    // Third: merge the predictions associated to those leaves.
    get_predictions_for_leaves(m_leafIndicesImage, m_relocaliserState->predictionsBlock, m_predictionsImage);

    // Finally: perform RANSAC.
    boost::optional<PoseCandidate> poseCandidate = m_preemptiveRansac->estimate_pose(m_keypointsImage, m_predictionsImage);

    // If we succeeded, grab the transformation matrix, fill the SE3Pose and return a GOOD relocalisation result.
    // We do this for the first m_maxRelocalisationsToOutput candidates estimated by P-RANSAC.
    if(poseCandidate)
    {
      Result result;
      result.pose.SetInvM(poseCandidate->cameraPose);
      result.quality = RELOCALISATION_GOOD;
      result.score = poseCandidate->energy;

      results.push_back(result);

      // We have to get the remaining best poses from the RANSAC pipeline.
      // We do this only if needed, to avoid needlessly copying data.
      if(m_maxRelocalisationsToOutput > 1)
      {
        std::vector<PoseCandidate> candidates;
        m_preemptiveRansac->get_best_poses(candidates);

        // Copy the best results in the output vector (skipping the first one,
        // since it's the same returned by m_preemptiveRansac->estimate_pose above).
        const size_t maxElements = std::min<size_t>(candidates.size(), m_maxRelocalisationsToOutput);
        for(size_t i = 1; i < maxElements; ++i)
        {
          Result result;
          result.pose.SetInvM(candidates[i].cameraPose);
          result.quality = RELOCALISATION_GOOD;
          result.score = candidates[i].energy;

          results.push_back(result);
        }
      }
    }
  }

  return results;
}

void ScoreRelocaliser::reset()
{
  // Set up the reservoirs if they haven't been allocated yet.
  if(!m_relocaliserState->exampleReservoirs)
  {
    m_relocaliserState->exampleReservoirs = ExampleReservoirsFactory<ExampleType>::make_reservoirs(m_reservoirCount, m_reservoirCapacity, m_deviceType, m_rngSeed);
  }

  // Set up the predictions block if it hasn't been allocated yet.
  if(!m_relocaliserState->predictionsBlock)
  {
    m_relocaliserState->predictionsBlock = MemoryBlockFactory::instance().make_block<ScorePrediction>(m_reservoirCount);
  }

  m_relocaliserState->exampleReservoirs->reset();
  m_relocaliserState->lastFeaturesAddedStartIdx = 0;
  m_relocaliserState->predictionsBlock->Clear();
  m_relocaliserState->reservoirUpdateStartIdx = 0;
}

void ScoreRelocaliser::save_to_disk(const std::string& outputFolder) const
{
  // First make sure that the output folder exists.
  bf::create_directories(outputFolder);

  // Then save the relocaliser's internal state to disk.
  m_relocaliserState->save_to_disk(outputFolder);
}

void ScoreRelocaliser::set_relocaliser_state(const ScoreRelocaliserState_Ptr& relocaliserState)
{
  m_relocaliserState = relocaliserState;
}

void ScoreRelocaliser::train(const ITMUChar4Image *colourImage, const ITMFloatImage *depthImage,
                             const Vector4f& depthIntrinsics, const ORUtils::SE3Pose& cameraPose)
{
  if(!m_relocaliserState->exampleReservoirs)
  {
    throw std::runtime_error("finish_training() has been called, cannot train the relocaliser until reset() is called.");
  }

  // First: select keypoints and compute descriptors.
  const Matrix4f invCameraPose = cameraPose.GetInvM();
  m_featureCalculator->compute_keypoints_and_features(colourImage, depthImage, invCameraPose, depthIntrinsics, m_keypointsImage.get(), m_descriptorsImage.get());

  // Second: find the leaves associated to the keypoints.
  m_scoreForest->find_leaves(m_descriptorsImage, m_leafIndicesImage);

  // Third: add keypoints to the correct reservoirs.
  m_relocaliserState->exampleReservoirs->add_examples(m_keypointsImage, m_leafIndicesImage);

  // Fourth: cluster some of the reservoirs.
  const uint32_t updateCount = compute_nb_reservoirs_to_update();
  m_exampleClusterer->cluster_examples(
    m_relocaliserState->exampleReservoirs->get_reservoirs(), m_relocaliserState->exampleReservoirs->get_reservoir_sizes(),
    m_relocaliserState->reservoirUpdateStartIdx, updateCount, m_relocaliserState->predictionsBlock
  );

  // Fifth: save the current index to indicate that reservoirs up to such index have to be clustered to represent the
  // examples that have just been added.
  m_relocaliserState->lastFeaturesAddedStartIdx = m_relocaliserState->reservoirUpdateStartIdx;

  // Finally: update starting index for the next invocation of either this function or idle_update().
  update_reservoir_start_idx();
}

void ScoreRelocaliser::update()
{
  if(!m_relocaliserState->exampleReservoirs)
  {
    throw std::runtime_error("finish_training() has been called, cannot update the relocaliser until reset() is called.");
  }

  // We are back to the first reservoir that was updated when
  // the last batch of features were added to the forest.
  // No need to perform further updates, we would get the same modes.
  // This check works only if the m_maxReservoirsToUpdate quantity
  // remains constant throughout the whole program.
  if(m_relocaliserState->reservoirUpdateStartIdx == m_relocaliserState->lastFeaturesAddedStartIdx) return;

  const uint32_t updateCount = compute_nb_reservoirs_to_update();
  m_exampleClusterer->cluster_examples(
    m_relocaliserState->exampleReservoirs->get_reservoirs(), m_relocaliserState->exampleReservoirs->get_reservoir_sizes(),
    m_relocaliserState->reservoirUpdateStartIdx, updateCount, m_relocaliserState->predictionsBlock
  );

  update_reservoir_start_idx();
}

void ScoreRelocaliser::update_all_clusters()
{
  // Simply call update until we get back to the first reservoir that hadn't been yet updated after the last call to train() was performed.
  while(m_relocaliserState->reservoirUpdateStartIdx != m_relocaliserState->lastFeaturesAddedStartIdx)
  {
    update();
  }
}

//#################### PRIVATE MEMBER FUNCTIONS ####################

uint32_t ScoreRelocaliser::compute_nb_reservoirs_to_update() const
{
  // Either the standard number of reservoirs to update or the remaining group until the end of the memory block.
  return std::min(m_maxReservoirsToUpdate, m_reservoirCount - m_relocaliserState->reservoirUpdateStartIdx);
}

void ScoreRelocaliser::ensure_valid_leaf(uint32_t treeIdx, uint32_t leafIdx) const
{
  if(treeIdx >= m_scoreForest->get_nb_trees() || leafIdx >= m_scoreForest->get_nb_leaves_in_tree(treeIdx))
  {
    throw std::invalid_argument("Error: Invalid tree or leaf index");
  }
}

void ScoreRelocaliser::update_reservoir_start_idx()
{
  m_relocaliserState->reservoirUpdateStartIdx += m_maxReservoirsToUpdate;

  // Restart from the first reservoir.
  if(m_relocaliserState->reservoirUpdateStartIdx >= m_reservoirCount) m_relocaliserState->reservoirUpdateStartIdx = 0;
}

}
