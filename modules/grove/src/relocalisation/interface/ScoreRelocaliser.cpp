/**
 * grove: ScoreRelocaliser.cpp
 * Copyright (c) Torr Vision Group, University of Oxford, 2017. All rights reserved.
 */

#include "relocalisation/interface/ScoreRelocaliser.h"

#include <boost/filesystem.hpp>
namespace bf = boost::filesystem;

#include <itmx/base/MemoryBlockFactory.h>
using namespace itmx;

#include <tvgutil/misc/SettingsContainer.h>
using namespace tvgutil;

namespace grove {

//#################### CONSTRUCTORS ####################

ScoreRelocaliser::ScoreRelocaliser(const SettingsContainer_CPtr& settings, const std::string& forestFilename)
  : m_settings(settings)
{
  const std::string settingsNamespace = "ScoreRelocaliser.";

  // In this constructor we are just setting the variables, instantiation of the sub-algorithms is left to the sub class
  // in order to instantiate the appropriate version.

  //
  // Relocaliser parameters
  //
  m_maxRelocalisationsToOutput = m_settings->get_first_value<uint32_t>(settingsNamespace + "maxRelocalisationsToOutput", 1);

  //
  // Forest
  //
  m_forestFilename = forestFilename;

  //
  // Reservoirs parameters
  //

  // Update the modes associated to this number of reservoirs for each integration/update call.
  m_maxReservoirsToUpdate = m_settings->get_first_value<uint32_t>(settingsNamespace + "maxReservoirsToUpdate", 256);
  // m_reservoirsCount is not set since that number depends on the forest that will be instantiated in the subclass.
  m_reservoirCapacity = m_settings->get_first_value<uint32_t>(settingsNamespace + "reservoirCapacity", 1024);
  m_rngSeed = m_settings->get_first_value<uint32_t>(settingsNamespace + "rngSeed", 42);

  //
  // Clustering parameters (defaults are tentative values that seem to work)
  //
  m_clustererSigma = m_settings->get_first_value<float>(settingsNamespace + "clustererSigma", 0.1f);
  m_clustererTau = m_settings->get_first_value<float>(settingsNamespace + "clustererTau", 0.05f);
  m_maxClusterCount = m_settings->get_first_value<uint32_t>(settingsNamespace + "maxClusterCount", ScorePrediction::Capacity);
  m_minClusterSize = m_settings->get_first_value<uint32_t>(settingsNamespace + "minClusterSize", 20);

  if(m_maxClusterCount > ScorePrediction::Capacity)
  {
    throw std::invalid_argument(settingsNamespace + "maxClusterCount > ScorePrediction::Capacity");
  }

  //
  // Relocaliser state.
  // (sets up an empty relocaliser state, with the assumption that the concrete subclasses will fill it with the right-sized variables).
  //
  m_relocaliserState.reset(new ScoreRelocaliserState);


  MemoryBlockFactory &mbf = MemoryBlockFactory::instance();

  // Setup memory blocks/images (except m_predictionsBlock since its size depends on the forest)
  m_leafIndicesImage = mbf.make_image<LeafIndices>();
  m_predictionsImage = mbf.make_image<ScorePrediction>();
  m_rgbdPatchDescriptorImage = mbf.make_image<DescriptorType>();
  m_rgbdPatchKeypointsImage = mbf.make_image<ExampleType>();
}

//#################### DESTRUCTOR ####################

ScoreRelocaliser::~ScoreRelocaliser() {}

//#################### PUBLIC MEMBER FUNCTIONS ####################

void ScoreRelocaliser::finish_training()
{
  boost::lock_guard<boost::recursive_mutex> lock(m_mutex);

  // First, update all clusters.
  update_all_clusters();

  // Now kill the contents of the reservoirs since we won't need them anymore.
  m_relocaliserState->exampleReservoirs.reset();
  m_relocaliserState->lastFeaturesAddedStartIdx = 0;
  m_relocaliserState->reservoirUpdateStartIdx = 0;

  // Release the clusterer as well.
  m_exampleClusterer.reset();
}

void ScoreRelocaliser::get_best_poses(std::vector<PoseCandidate>& poseCandidates) const
{
  // Just forward the vector to P-RANSAC.
  m_preemptiveRansac->get_best_poses(poseCandidates);
}

Keypoint3DColourImage_CPtr ScoreRelocaliser::get_keypoints_image() const
{
  return m_rgbdPatchKeypointsImage;
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

void ScoreRelocaliser::load_from_disk(const std::string& inputFolder)
{
  // Load the relocaliser state from disk.
  m_relocaliserState->load_from_disk(inputFolder);
}

std::vector<Relocaliser::Result> ScoreRelocaliser::relocalise(const ITMUChar4Image *colourImage, const ITMFloatImage *depthImage, const Vector4f& depthIntrinsics) const
{
  boost::lock_guard<boost::recursive_mutex> lock(m_mutex);

  std::vector<Result> results;

  // Try to estimate a pose only if we have enough valid depth values.
  if(m_lowLevelEngine->CountValidDepths(depthImage) > m_preemptiveRansac->get_min_nb_required_points())
  {
    // First: select keypoints and compute descriptors.
    m_featureCalculator->compute_keypoints_and_features(colourImage, depthImage, depthIntrinsics, m_rgbdPatchKeypointsImage.get(), m_rgbdPatchDescriptorImage.get());

    // Second: find all the leaves associated to the keypoints.
    m_scoreForest->find_leaves(m_rgbdPatchDescriptorImage, m_leafIndicesImage);

    // Third: merge the predictions associated to those leaves.
    get_predictions_for_leaves(m_leafIndicesImage, m_relocaliserState->predictionsBlock, m_predictionsImage);

    // Finally: perform RANSAC.
    boost::optional<PoseCandidate> poseCandidate = m_preemptiveRansac->estimate_pose(m_rgbdPatchKeypointsImage, m_predictionsImage);

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
  boost::lock_guard<boost::recursive_mutex> lock(m_mutex);

  m_relocaliserState->exampleReservoirs->reset();
  m_relocaliserState->predictionsBlock->Clear();

  m_relocaliserState->lastFeaturesAddedStartIdx = 0;
  m_relocaliserState->reservoirUpdateStartIdx = 0;
}

void ScoreRelocaliser::save_to_disk(const std::string& outputFolder) const
{
  // First, make sure the output folder exists.
  bf::create_directories(outputFolder);

  // The serialise the data. Everything is contained in the relocaliser state, so that's the only thing we have to serialise.
  m_relocaliserState->save_to_disk(outputFolder);
}

void ScoreRelocaliser::set_relocaliser_state(const ScoreRelocaliserState_Ptr& relocaliserState)
{
  m_relocaliserState = relocaliserState;
}

void ScoreRelocaliser::train(const ITMUChar4Image *colourImage, const ITMFloatImage *depthImage,
                             const Vector4f& depthIntrinsics, const ORUtils::SE3Pose& cameraPose)
{
  boost::lock_guard<boost::recursive_mutex> lock(m_mutex);

  if(!m_relocaliserState->exampleReservoirs)
  {
    throw std::runtime_error("finish_training() has been called, cannot train the relocaliser until reset() is called.");
  }

  // First: select keypoints and compute descriptors.
  const Matrix4f invCameraPose = cameraPose.GetInvM();
  m_featureCalculator->compute_keypoints_and_features(colourImage, depthImage, invCameraPose, depthIntrinsics, m_rgbdPatchKeypointsImage.get(), m_rgbdPatchDescriptorImage.get());

  // Second: find the leaves associated to the keypoints.
  m_scoreForest->find_leaves(m_rgbdPatchDescriptorImage, m_leafIndicesImage);

  // Third: add keypoints to the correct reservoirs.
  m_relocaliserState->exampleReservoirs->add_examples(m_rgbdPatchKeypointsImage, m_leafIndicesImage);

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
  boost::lock_guard<boost::recursive_mutex> lock(m_mutex);

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
  boost::lock_guard<boost::recursive_mutex> lock(m_mutex);

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
  return std::min(m_maxReservoirsToUpdate, m_reservoirsCount - m_relocaliserState->reservoirUpdateStartIdx);
}

void ScoreRelocaliser::update_reservoir_start_idx()
{
  m_relocaliserState->reservoirUpdateStartIdx += m_maxReservoirsToUpdate;

  // Restart from the first reservoir.
  if(m_relocaliserState->reservoirUpdateStartIdx >= m_reservoirsCount) m_relocaliserState->reservoirUpdateStartIdx = 0;
}

}
