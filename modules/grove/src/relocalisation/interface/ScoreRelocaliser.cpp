/**
 * grove: ScoreRelocaliser.cpp
 * Copyright (c) Torr Vision Group, University of Oxford, 2017. All rights reserved.
 */

#include "relocalisation/interface/ScoreRelocaliser.h"
using namespace ORUtils;

#include <boost/filesystem.hpp>
namespace bf = boost::filesystem;

#ifdef WITH_OPENCV
#include <opencv2/opencv.hpp>
#endif

#include <orx/base/MemoryBlockFactory.h>
using namespace orx;

#include <tvgutil/misc/SettingsContainer.h>
using namespace tvgutil;

#include "clustering/ExampleClustererFactory.h"
#include "features/FeatureCalculatorFactory.h"
#include "forests/DecisionForestFactory.h"
#include "ransac/PreemptiveRansacFactory.h"
#include "reservoirs/ExampleReservoirsFactory.h"


#include <fstream>
#include <Eigen/Geometry>
bool b_ours_rel = false; // if use our deep octree or random forests.
bool b_confwei_rel = true; // if we use confidence or not.
char g_seq_name_rel[50] = "test_cpv1_rio_home_tree1_gmm2"; // files.
bool b_debug_rel = false; // for debug.
int g_frame_max_rel = 10; // for debug.
int n_nbInliers_rel = 100; // fixed. default 100.
int g_frame_id_rel = 1; // fixed.
// test5x_gmm_v1.0_confwei 
// test5x_cpv2_gmm_v0.0_confwei 
// train_cpv2_tree1_gmm2_ub_level3
// test5x_cpv1_tree2_gmm2

namespace grove {

//#################### CONSTRUCTORS ####################

ScoreRelocaliser::ScoreRelocaliser(const std::string& forestFilename, const SettingsContainer_CPtr& settings, const std::string& settingsNamespace, DeviceType deviceType)
: m_backed(false),
  m_deviceType(deviceType),
  m_maxX(static_cast<float>(INT_MIN)),
  m_maxY(static_cast<float>(INT_MIN)),
  m_maxZ(static_cast<float>(INT_MIN)),
  m_minX(static_cast<float>(INT_MAX)),
  m_minY(static_cast<float>(INT_MAX)),
  m_minZ(static_cast<float>(INT_MAX)),
  m_settings(settings)
{
  // Determine the top-level parameters for the relocaliser.
  m_maxRelocalisationsToOutput = m_settings->get_first_value<uint32_t>(settingsNamespace + "maxRelocalisationsToOutput", 1);
  m_visualiseForest = m_settings->get_first_value<bool>(settingsNamespace + "visualiseForest", false);

  // Determine the reservoir-related parameters.
  m_maxReservoirsToUpdate = m_settings->get_first_value<uint32_t>(settingsNamespace + "maxReservoirsToUpdate", 256);  // Update the modes associated with this number of reservoirs for each train/update call.
  m_reservoirCapacity = m_settings->get_first_value<uint32_t>(settingsNamespace + "reservoirCapacity", 1024);
  m_rngSeed = m_settings->get_first_value<uint32_t>(settingsNamespace + "rngSeed", 42);

  // Determine the clustering-related parameters (the defaults are tentative values that seem to work).
  m_clustererSigma = m_settings->get_first_value<float>(settingsNamespace + "clustererSigma", 0.1f);
  m_clustererTau = m_settings->get_first_value<float>(settingsNamespace + "clustererTau", 0.05f); // raw
  //m_clustererTau = m_settings->get_first_value<float>(settingsNamespace + "clustererTau", 0.10f); 
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

  // Instantiate the sub-components.
  m_featureCalculator = FeatureCalculatorFactory::make_da_rgbd_patch_feature_calculator(deviceType);
  m_preemptiveRansac = PreemptiveRansacFactory::make_preemptive_ransac(settings, settingsNamespace + "PreemptiveRansac.", deviceType);

  m_scoreForest = m_settings->get_first_value<bool>(settingsNamespace + "randomlyGenerateForest", false)
    ? DecisionForestFactory<DescriptorType,FOREST_TREE_COUNT>::make_randomly_generated_forest(m_settings, deviceType)
    : DecisionForestFactory<DescriptorType,FOREST_TREE_COUNT>::make_forest(forestFilename, deviceType);

  m_reservoirCount = m_scoreForest->get_nb_leaves();

  // Set up the relocaliser's internal state.
  m_relocaliserState.reset(new ScoreRelocaliserState);
  reset();
}

//#################### DESTRUCTOR ####################

ScoreRelocaliser::~ScoreRelocaliser() {}

//#################### PUBLIC MEMBER FUNCTIONS ####################

void ScoreRelocaliser::finish_training()
{
  // If this relocaliser is "backed" by another one, early out.
  if(m_backed) return;

  boost::lock_guard<boost::recursive_mutex> lock(m_mutex);

  // First update all of the clusters.
  update_all_clusters();

  // Then kill the contents of the reservoirs (we won't need them any more).
  m_relocaliserState->exampleReservoirs.reset();
  m_relocaliserState->lastExamplesAddedStartIdx = 0;
  m_relocaliserState->reservoirUpdateStartIdx = 0;

  // Finally, release the example clusterer.
  m_exampleClusterer.reset();
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

ORUChar4Image_CPtr ScoreRelocaliser::get_visualisation_image(const std::string& key) const
{
  if(key == "leaves") return m_pixelsToLeavesImage;
  else if(key == "points") return m_pixelsToPointsImage;
  else return ORUChar4Image_CPtr();
}

void ScoreRelocaliser::load_from_disk(const std::string& inputFolder)
{
  // If this relocaliser is "backed" by another one, early out.
  if(m_backed) return;

  // Otherwise, load its internal state from disk.
  m_relocaliserState->load_from_disk(inputFolder);
}

std::vector<Relocaliser::Result> ScoreRelocaliser::relocalise_rel(const ORUChar4Image *colourImage, const ORFloatImage *depthImage, const Vector4f& depthIntrinsics, int fid)
{
  boost::lock_guard<boost::recursive_mutex> lock(m_mutex);

  std::vector<Result> results;

  // Iff we have enough valid depth values, try to estimate the camera pose:
  if(count_valid_depths(depthImage) > m_preemptiveRansac->get_min_nb_required_points())
  {
    // Step 1: Extract keypoints from the RGB-D image and compute descriptors for them.
    m_featureCalculator->compute_keypoints_and_features(colourImage, depthImage, depthIntrinsics, m_keypointsImage.get(), m_descriptorsImage.get());

    // Step 2: Find all of the leaves in the forest that are associated with the descriptors for the keypoints.
    m_scoreForest->find_leaves(m_descriptorsImage, m_leafIndicesImage);

    // Step 3: Merge the SCoRe predictions (sets of clusters) associated with each keypoint to create a single
    //         SCoRe prediction (a single set of clusters) for each keypoint.
    merge_predictions_for_keypoints(m_leafIndicesImage, m_predictionsImage);


    // Step 4: Perform P-RANSAC to try to estimate the camera pose.
    boost::optional<PoseCandidate> poseCandidate = m_preemptiveRansac->estimate_pose(m_keypointsImage, m_predictionsImage);

    // Step 5: If we succeeded in estimated a camera pose:
    if(poseCandidate)
    {
      // Add the pose to the results.
      Result result;
      result.pose.SetInvM(poseCandidate->cameraPose);
      result.quality = RELOCALISATION_GOOD;
      result.score = poseCandidate->energy;
      results.push_back(result);

      // If we're outputting multiple poses:
      if(m_maxRelocalisationsToOutput > 1)
      {
        // Get all of the candidates that survived the initial culling process during P-RANSAC.
        std::vector<PoseCandidate> candidates;
        m_preemptiveRansac->get_best_poses(candidates);

        // Add the best candidates to the results (skipping the first one, since it's the same one returned by estimate_pose above).
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

  // If forest visualisation is enabled and we relocalised successfully, update the forest visualisation images (for debugging purposes).
  if(m_visualiseForest && !results.empty())
  {
    update_pixels_to_leaves_image(depthImage);

    // Note: We use the "best" pose here as a default, even though this may later be either refined by ICP or discarded in favour of a different pose.
    update_pixels_to_points_image(results[0].pose);
  }

  return results;
}

//_CPU_AND_GPU_CODE_ 
std::vector<Relocaliser::Result> ScoreRelocaliser::relocalise(const ORUChar4Image *colourImage, const ORFloatImage *depthImage, const Vector4f& depthIntrinsics) const
{
  boost::lock_guard<boost::recursive_mutex> lock(m_mutex);

  std::vector<Result> results;

  // Iff we have enough valid depth values, try to estimate the camera pose:
  if(count_valid_depths(depthImage) > m_preemptiveRansac->get_min_nb_required_points())
  {
    // Step 1: Extract keypoints from the RGB-D image and compute descriptors for them.
    m_featureCalculator->compute_keypoints_and_features(colourImage, depthImage, depthIntrinsics, m_keypointsImage.get(), m_descriptorsImage.get());

    // Step 2: Find all of the leaves in the forest that are associated with the descriptors for the keypoints.
    m_scoreForest->find_leaves(m_descriptorsImage, m_leafIndicesImage);

    // Step 3: Merge the SCoRe predictions (sets of clusters) associated with each keypoint to create a single
    //         SCoRe prediction (a single set of clusters) for each keypoint.
    merge_predictions_for_keypoints(m_leafIndicesImage, m_predictionsImage);

    // debug.
    if (b_debug_rel)
    {
        if (g_frame_id_rel>g_frame_max_rel)
      {
        printf("currently %d frames done. continue?\n", g_frame_id_rel);
        getchar();
      }
    }//*/


    // deep octree
    bool b_valid_prediction = true;
    if (b_ours_rel)
    {
      // ablate colour.
      {
        m_keypointsImage->UpdateHostFromDevice();
        m_predictionsImage->UpdateHostFromDevice();
        Keypoint3DColour* keypoints = m_keypointsImage->GetData(MEMORYDEVICE_CPU);
        auto predictions = m_predictionsImage->GetData(MEMORYDEVICE_CPU);
        for (int x = 0; x < 56; ++x)
        {
          for (int y = 0; y < 43; ++y)
          {
            int idx = y * 56 + x;
            keypoints[idx].colour[0] = 0;
            keypoints[idx].colour[1] = 0;
            keypoints[idx].colour[2] = 0;
            for (int cid = 0; cid < predictions[idx].size; ++cid)
            {
              predictions[idx].elts[cid].colour[0] = 0;
              predictions[idx].elts[cid].colour[1] = 0;
              predictions[idx].elts[cid].colour[2] = 0;
            }
          }
        }
        m_keypointsImage->UpdateDeviceFromHost();
        m_predictionsImage->UpdateDeviceFromHost();
      }
      // trans from GPU to CPU
      m_keypointsImage->UpdateHostFromDevice();
      m_predictionsImage->UpdateHostFromDevice();
      auto predictions = m_predictionsImage->GetData(MEMORYDEVICE_CPU);
      auto keypoints = m_keypointsImage->GetData(MEMORYDEVICE_CPU);
      for (int idx = 0; idx < 56*43; ++idx)
      {
        keypoints[idx].valid = false;
      }
      // load from file
      {
        char predPath[200];
        std::sprintf(predPath, "/home/DeepOctree/%s/predictionsImage_%d.txt", g_seq_name_rel, g_frame_id_rel);
        std::ifstream ifs(predPath);
        int num_valid = 0;
        ifs >> num_valid;
        if (num_valid<5) b_valid_prediction = false;
        //printf("%d valid predictions\n", num_valid);
        for (int pid = 0; pid < num_valid; ++pid)
        {
          float confidence = 0;
          int idx = 0;
          ifs >> idx;
          keypoints[idx].valid = true; // - - - set valid true - - -
          if (b_confwei_rel) ifs >> confidence;
          int n_cluster = 0;
          ifs >> n_cluster;

          // replace clusters
          {
            predictions[idx].size = 0;
            for (int cid = 0; cid < n_cluster; ++cid)
            {
              int r, g, b;
              float x, y ,z;
              float weight = 0;
              if (b_confwei_rel) ifs >> weight;
              ifs >> r >> g >> b >> x >> y >> z;
              float x00, x01, x02, x10, x11, x12, x20, x21, x22;
              ifs >> x00 >> x01 >> x02 >> x10 >> x11 >> x12 >> x20 >> x21 >> x22;
              predictions[idx].elts[cid].colour = Vector3u(r, g, b);
              predictions[idx].elts[cid].position = Vector3f(x, y, z);
              predictions[idx].elts[cid].nbInliers = (int)(n_nbInliers_rel/n_cluster);
              if (b_confwei_rel) predictions[idx].elts[cid].nbInliers = (int)ceil(predictions[idx].elts[cid].nbInliers*confidence*100);
              predictions[idx].elts[cid].positionInvCovariance(0,0) = x00;
              predictions[idx].elts[cid].positionInvCovariance(0,1) = x01;
              predictions[idx].elts[cid].positionInvCovariance(0,2) = x02;
              predictions[idx].elts[cid].positionInvCovariance(1,0) = x10;
              predictions[idx].elts[cid].positionInvCovariance(1,1) = x11;
              predictions[idx].elts[cid].positionInvCovariance(1,2) = x12;
              predictions[idx].elts[cid].positionInvCovariance(2,0) = x20;
              predictions[idx].elts[cid].positionInvCovariance(2,1) = x21;
              predictions[idx].elts[cid].positionInvCovariance(2,2) = x22;
              predictions[idx].size++;
            }
          }
          
          // // replace clusters
          // {
          //   int delta = predictions[idx].size/n_cluster;
          //   if (delta==0) delta = 1;
          //   int beg_id = 0;
          //   int end_id = beg_id + delta;
          //   for (int cid = 0; cid < n_cluster; ++cid)
          //   {
          //     int r, g, b;
          //     float x, y ,z;
          //     float weight = 0;
          //     if (b_confwei_rel) ifs >> weight;
          //     ifs >> r >> g >> b >> x >> y >> z;
          //     float x00, x01, x02, x10, x11, x12, x20, x21, x22;
          //     ifs >> x00 >> x01 >> x02 >> x10 >> x11 >> x12 >> x20 >> x21 >> x22;
          //     for (int raw_id = beg_id; raw_id < end_id; ++raw_id)
          //     {
          //       predictions[idx].elts[raw_id].colour = Vector3u(r, g, b);
          //       predictions[idx].elts[raw_id].position = Vector3f(x, y, z);
          //       predictions[idx].elts[raw_id].nbInliers = (int)(n_nbInliers_rel/n_cluster);
          //       //if (b_confwei_rel) predictions[idx].elts[raw_id].nbInliers = (int)ceil(predictions[idx].elts[raw_id].nbInliers*confidence*weight*10000);
          //       if (b_confwei_rel) predictions[idx].elts[raw_id].nbInliers = (int)ceil(predictions[idx].elts[raw_id].nbInliers*confidence*100);
          //       //if (b_confwei_rel) predictions[idx].elts[raw_id].nbInliers = (int)ceil(predictions[idx].elts[raw_id].nbInliers*weight*100);
          //       //if (b_confwei_rel) predictions[idx].elts[raw_id].nbInliers = (int)ceil(predictions[idx].elts[raw_id].nbInliers*confidence*confidence*10000);
          //       predictions[idx].elts[raw_id].positionInvCovariance(0,0) = x00;
          //       predictions[idx].elts[raw_id].positionInvCovariance(0,1) = x01;
          //       predictions[idx].elts[raw_id].positionInvCovariance(0,2) = x02;
          //       predictions[idx].elts[raw_id].positionInvCovariance(1,0) = x10;
          //       predictions[idx].elts[raw_id].positionInvCovariance(1,1) = x11;
          //       predictions[idx].elts[raw_id].positionInvCovariance(1,2) = x12;
          //       predictions[idx].elts[raw_id].positionInvCovariance(2,0) = x20;
          //       predictions[idx].elts[raw_id].positionInvCovariance(2,1) = x21;
          //       predictions[idx].elts[raw_id].positionInvCovariance(2,2) = x22;
          //     }
          //     beg_id += delta;
          //     end_id += delta;
          //   }
          //   predictions[idx].size = beg_id;
          // }

        }
        ifs.close();
      }
      // trans from CPU to GPU
      m_keypointsImage->UpdateDeviceFromHost();
      m_predictionsImage->UpdateDeviceFromHost();
    }//*/

    // Step 4: Perform P-RANSAC to try to estimate the camera pose.
    boost::optional<PoseCandidate> poseCandidate;
    if (b_valid_prediction) poseCandidate = m_preemptiveRansac->estimate_pose(m_keypointsImage, m_predictionsImage);
    g_frame_id_rel++; 

    // Step 5: If we succeeded in estimated a camera pose:
    if(poseCandidate)
    {
      // Add the pose to the results.
      Result result;
      result.pose.SetInvM(poseCandidate->cameraPose);
      result.quality = RELOCALISATION_GOOD;
      result.score = poseCandidate->energy;
      results.push_back(result);

      // If we're outputting multiple poses:
      if(m_maxRelocalisationsToOutput > 1)
      {
        // Get all of the candidates that survived the initial culling process during P-RANSAC.
        std::vector<PoseCandidate> candidates;
        m_preemptiveRansac->get_best_poses(candidates);

        // Add the best candidates to the results (skipping the first one, since it's the same one returned by estimate_pose above).
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

  // If forest visualisation is enabled and we relocalised successfully, update the forest visualisation images (for debugging purposes).
  if(m_visualiseForest && !results.empty())
  {
    update_pixels_to_leaves_image(depthImage);

    // Note: We use the "best" pose here as a default, even though this may later be either refined by ICP or discarded in favour of a different pose.
    update_pixels_to_points_image(results[0].pose);
  }

  return results;
}

void ScoreRelocaliser::reset()
{
  // If this relocaliser is "backed" by another one, early out.
  if(m_backed) return;

  boost::lock_guard<boost::recursive_mutex> lock(m_mutex);

  // Set up the clusterer if it hasn't been allocated yet.
  if(!m_exampleClusterer)
  {
    m_exampleClusterer = ExampleClustererFactory<ExampleType,ClusterType,PredictionType::Capacity>::make_clusterer(
      m_clustererSigma, m_clustererTau, m_maxClusterCount, m_minClusterSize, m_deviceType
    );
  }

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
  m_relocaliserState->lastExamplesAddedStartIdx = 0;
  m_relocaliserState->predictionsBlock->Clear();
  m_relocaliserState->reservoirUpdateStartIdx = 0;
}

void ScoreRelocaliser::save_to_disk(const std::string& outputFolder) const
{
  // If this relocaliser is "backed" by another one, early out.
  if(m_backed) return;

  // First make sure that the output folder exists.
  bf::create_directories(outputFolder);

  // Then save the relocaliser's internal state to disk.
  m_relocaliserState->save_to_disk(outputFolder);
}

void ScoreRelocaliser::set_backing_relocaliser(const ScoreRelocaliser_Ptr& backingRelocaliser)
{
  m_relocaliserState = backingRelocaliser->m_relocaliserState;
  m_backed = true;
}

void ScoreRelocaliser::train(const ORUChar4Image *colourImage, const ORFloatImage *depthImage,
                             const Vector4f& depthIntrinsics, const ORUtils::SE3Pose& cameraPose)
{

  boost::lock_guard<boost::recursive_mutex> lock(m_mutex);

  // If forest visualisation is enabled, update the maximum and minimum x, y and z coordinates visited by the camera during training.
  if(m_visualiseForest)
  {
    m_maxX = std::max(m_maxX, cameraPose.GetT().x);
    m_maxY = std::max(m_maxY, cameraPose.GetT().y);
    m_maxZ = std::max(m_maxZ, cameraPose.GetT().z);
    m_minX = std::min(m_minX, cameraPose.GetT().x);
    m_minY = std::min(m_minY, cameraPose.GetT().y);
    m_minZ = std::min(m_minZ, cameraPose.GetT().z);
  }

  // If this relocaliser is "backed" by another one, early out.
  if(m_backed) return;

  if(!m_relocaliserState->exampleReservoirs)
  {
    throw std::runtime_error("Error: finish_training() has been called; the relocaliser cannot be trained again until reset() is called");
  }

  // Step 1: Extract keypoints from the RGB-D image and compute descriptors for them.
  const Matrix4f invCameraPose = cameraPose.GetInvM();
  m_featureCalculator->compute_keypoints_and_features(colourImage, depthImage, invCameraPose, depthIntrinsics, m_keypointsImage.get(), m_descriptorsImage.get());

  // Step 2: Find all of the leaves in the forest that are associated with the descriptors for the keypoints.
  m_scoreForest->find_leaves(m_descriptorsImage, m_leafIndicesImage);

  // Step 3: Add the keypoints to the relevant reservoirs.
  m_relocaliserState->exampleReservoirs->add_examples(m_keypointsImage, m_leafIndicesImage);

  // Step 4: Cluster some of the reservoirs.
  const uint32_t nbReservoirsToUpdate = compute_nb_reservoirs_to_update();
  m_exampleClusterer->cluster_examples(
    m_relocaliserState->exampleReservoirs->get_reservoirs(), m_relocaliserState->exampleReservoirs->get_reservoir_sizes(),
    m_relocaliserState->reservoirUpdateStartIdx, nbReservoirsToUpdate, m_relocaliserState->predictionsBlock
  );

  // Step 5: Store the index of the first reservoir that was just updated so that we can tell when there are no more clusters to update.
  m_relocaliserState->lastExamplesAddedStartIdx = m_relocaliserState->reservoirUpdateStartIdx;

  // Step 6: Update the index of the first reservoir to subject to clustering during the next train/update call.
  update_reservoir_start_idx();
}


void ScoreRelocaliser::test4pcd(const ORUChar4Image *colourImage, const ORFloatImage *depthImage, const Vector4f& depthIntrinsics, const ORUtils::SE3Pose& cameraPose, CorrectPointCloud& pointCloud)
{
  
  boost::lock_guard<boost::recursive_mutex> lock(m_mutex);
  // Step 1: Get all the points from the RGB-D image and compute descriptors for them.
  m_featureCalculator->set_feature_step(1);
  m_featureCalculator->compute_keypoints_and_features(colourImage, depthImage, depthIntrinsics, m_keypointsImage.get(), m_descriptorsImage.get());

  // Step 2: Find all of the leaves in the forest that are associated with the descriptors for the keypoints.
  m_scoreForest->find_leaves(m_descriptorsImage, m_leafIndicesImage);

  // Merge the SCoRe predictions (sets of clusters) associated with each keypoint to create a single
  //         SCoRe prediction (a single set of clusters) for each keypoint.
  // merge_predictions_for_keypoints(m_leafIndicesImage, m_predictionsImage);

  // Step 3: Get the cloest distance, and judge whether these points are valid
  //         Ensure that the keypoints, leaf indices and ScoRe predictions are available on the CPU.
  m_keypointsImage->UpdateHostFromDevice();
  m_leafIndicesImage->UpdateHostFromDevice();
  // m_predictionsImage->UpdateHostFromDevice();
  depthImage->UpdateHostFromDevice();
  //         Use the reservoirs to get examples 
  const int ReservoirIndexCount = FOREST_TREE_COUNT;
  const ExampleType *reservoirs = m_relocaliserState->exampleReservoirs->get_reservoirs()->GetData(MEMORYDEVICE_CPU);
  const uint32_t reservoirCapacity = m_relocaliserState->exampleReservoirs->get_reservoir_capacity();

  //std::fstream outputFile("/Codes/xyz.txt", std::ios_base::out);
  for (int i = 0, pointsCount = static_cast<int>(m_keypointsImage->dataSize); i < pointsCount; ++i)
  {
    // If the pixel has a valid keypoint, look up the position of the cluster (if any) in the corresponding prediction that is closest to it.
    const ExampleType& keypoint = m_keypointsImage->GetData(MEMORYDEVICE_CPU)[i];
      
    if(!keypoint.valid) continue;

    // Get the leaf indice of this keypoint
    const int *reservoirIndices = m_leafIndicesImage->GetData(MEMORYDEVICE_CPU)[i].v;

    // Get the ground truth position
    const Vector3f keyPointPosition = cameraPose.GetInvM() * keypoint.position;
    bool found = false;

    // Try to find examples from each specified reservoir
    for (int j = 0; j < ReservoirIndexCount; j++)
    {
      // The reservoir index (this corresponds to a row in the reservoirs image).
      const int reservoirIdx = reservoirIndices[j];

      // The raster index (in the reservoirs image) of the first example in the reservoir.
      const int reservoirStartIdx = reservoirIdx * reservoirCapacity;
      
      // Judge the leaf nodes valid or not
      /*
      float count = 0;
      Vector3f sumV(0, 0, 0);
      for (int k = 0; k < reservoirCapacity; k++)
      {
        const ExampleType example = reservoirs[reservoirStartIdx + k];
        if (!example.valid) continue;
        const Vector3f examplePosition = example.position;
        sumV = sumV + examplePosition;
        count += 1;
      }
      if (count == 0 || count > 200) {
        continue;
      }
      sumV = sumV / count;
      float deviation = 0;
      for (int k = 0; k < reservoirCapacity; k++)
      {
        const ExampleType example = reservoirs[reservoirStartIdx + k];
        if (!example.valid) continue;
        const Vector3f examplePosition = example.position;
        Vector3f diff = sumV - examplePosition;
        diff *= diff;
        deviation += (diff[0] + diff[1] + diff[2]);
      }
      deviation /= count;
      deviation = std::sqrt(deviation); 
      if (deviation > 3.5)
      {
        continue;
      }
      */
      for (int k = 0; k < reservoirCapacity; k++)
      {
        // Get the example in the reservoir.
        const ExampleType example = reservoirs[reservoirStartIdx + k];
        if (!example.valid) continue;
        // Get the position of example point.
        const Vector3f examplePosition = example.position;
        // Calculate the distance
        const Vector3f diff = keyPointPosition - examplePosition;
        const Vector3f squareDiff = diff * diff;
        double Distance = std::sqrt(squareDiff.x + squareDiff.y + squareDiff.z);
        // If two points are close enough then store the point in correct point cloud.
        if (Distance < 5e-2)
        {   
          // Create a struct for the point.
          CorrectPointType point(25, 0);
          for (int p = 0; p < 3; p++) 
          {
            point[p] = double(int(keyPointPosition[p] * 100) / 5 * 5) / 100.0;
          }
          float pointDepth = depthImage->GetData(MEMORYDEVICE_CPU)[i];
          int distancePartition = int(pointDepth / 0.5);
          if (distancePartition >= 10) distancePartition = 9;

          point[3 + distancePartition] += 1;
	  point[23] += 1;
          //outputFile << reservoirStartIdx / reservoirCapacity << std::endl;

          // Put the point in results
          pointCloud.push_back(point);
          // Once a keypoint are seen as a correct point, there is no need to continue to judge it.
          found = true;
          break;
        }
        //std::cout << found << std::endl;
      }
      if (found)
	    {
	      break;
	    }
    }
    if (!found)
    {
      // Create a struct for the point.
      CorrectPointType point(25, 0);
      for (int p = 0; p < 3; p++) 
      {
        point[p] = double(int(keyPointPosition[p] * 100) / 5 * 5) / 100.0; 
      }
      float pointDepth = depthImage->GetData(MEMORYDEVICE_CPU)[i];

      int distancePartition = int(pointDepth / 0.5);
      if (distancePartition >= 10) distancePartition = 9;

      point[3 + distancePartition + 10] += 1;
      point[24] += 1;
      // Put the point in results
      pointCloud.push_back(point);
    }
  }
  /*
  //const ExampleType *reservoirs = m_relocaliserState->exampleReservoirs->get_reservoirs()->GetData(MEMORYDEVICE_CPU);
  //const uint32_t reservoirCapacity = m_relocaliserState->exampleReservoirs->get_reservoir_capacity();
  int start_idx = 347 * reservoirCapacity;
  for (int i = 0; i < reservoirCapacity; i++)
  {
	  const ExampleType example = reservoirs[start_idx + i];
      if (!example.valid) continue;
      const Vector3f examplePosition = example.position;
      outputFile << examplePosition.x << ";" << examplePosition.y << ";" << examplePosition.z << ";" << std::endl;
  }
  std::cout << reservoirCapacity << " " << m_relocaliserState->exampleReservoirs->get_reservoirs()->dataSize << std::endl;
  std::fstream outputFile("/Codes/rf.txt", std::ios_base::out);
  for (int i = 0; i < static_cast<int>(m_relocaliserState->exampleReservoirs->get_reservoirs()->dataSize); i += reservoirCapacity)
  {
    float count = 0;
    Vector3f sumV(0, 0, 0);
    for (int j = 0; j < reservoirCapacity; j++)
    {
      const ExampleType example = reservoirs[i + j];
      if (!example.valid) continue;
      const Vector3f examplePosition = example.position;
      sumV = sumV + examplePosition;
      count += 1;
    }
    if (count == 0) {
	    outputFile << "0\t0" << std::endl;
	    continue;
    }
    sumV = sumV / count;
    float variance = 0;
    for (int j = 0; j < reservoirCapacity; j++)
    {
      const ExampleType example = reservoirs[i + j];
      if (!example.valid) continue;
      const Vector3f examplePosition = example.position;
      Vector3f diff = sumV - examplePosition;
      diff *= diff;
      variance += (diff[0] + diff[1] + diff[2]);
    }
    variance /= count;
    outputFile << count << "\t" <<variance << std::endl;
  }
  std::cout << "end" << std::endl;
  */
  //outputFile.close();
}

void ScoreRelocaliser::update()
{
  // If this relocaliser is "backed" by another one, early out.
  if(m_backed) return;

  boost::lock_guard<boost::recursive_mutex> lock(m_mutex);

  if(!m_relocaliserState->exampleReservoirs)
  {
    throw std::runtime_error("Error: finish_training() has been called; the relocaliser cannot be updated again until reset() is called");
  }

  // If we are back to the first reservoir that was updated when the last batch of examples were added to the
  // forest, there is no need to perform further updates, since we would get the same clusters. Note that this
  // check only works if the m_maxReservoirsToUpdate quantity remains constant throughout the whole program.
  if(m_relocaliserState->reservoirUpdateStartIdx == m_relocaliserState->lastExamplesAddedStartIdx) return;

  // Otherwise, cluster the next batch of reservoirs, and update the index of the first reservoir to subject to
  // clustering during the next train/update call.
  const uint32_t updateCount = compute_nb_reservoirs_to_update();
  m_exampleClusterer->cluster_examples(
    m_relocaliserState->exampleReservoirs->get_reservoirs(), m_relocaliserState->exampleReservoirs->get_reservoir_sizes(),
    m_relocaliserState->reservoirUpdateStartIdx, updateCount, m_relocaliserState->predictionsBlock
  );

  update_reservoir_start_idx();
}

void ScoreRelocaliser::update_all_clusters()
{
  // If this relocaliser is "backed" by another one, early out.
  if(m_backed) return;

  boost::lock_guard<boost::recursive_mutex> lock(m_mutex);

  // Repeatedly call update until we get back to the batch of reservoirs that was updated last time train() was called.
  while(m_relocaliserState->reservoirUpdateStartIdx != m_relocaliserState->lastExamplesAddedStartIdx)
  {
    update();
  }
}

//#################### PRIVATE MEMBER FUNCTIONS ####################

uint32_t ScoreRelocaliser::compute_nb_reservoirs_to_update() const
{
  // Either the standard number of reservoirs to update, or the number remaining before the end of the memory block.
  return std::min(m_maxReservoirsToUpdate, m_reservoirCount - m_relocaliserState->reservoirUpdateStartIdx);
}

void ScoreRelocaliser::ensure_valid_leaf(uint32_t treeIdx, uint32_t leafIdx) const
{
  if(treeIdx >= m_scoreForest->get_nb_trees() || leafIdx >= m_scoreForest->get_nb_leaves_in_tree(treeIdx))
  {
    throw std::invalid_argument("Error: Invalid tree or leaf index");
  }
}

void ScoreRelocaliser::update_pixels_to_leaves_image(const ORFloatImage *depthImage) const
{
#ifdef WITH_OPENCV
  // Ensure that the depth image and leaf indices are available on the CPU.
  depthImage->UpdateHostFromDevice();
  m_leafIndicesImage->UpdateHostFromDevice();

  // Make a map showing which pixels are in which leaves (for the first tree).
  std::map<int,std::vector<int> > leafToRegionMap;
  for(int i = 0, pixelCount = static_cast<int>(m_leafIndicesImage->dataSize); i < pixelCount; ++i)
  {
    const ORUtils::VectorX<int,ScoreRelocaliser::FOREST_TREE_COUNT>& elt = m_leafIndicesImage->GetData(MEMORYDEVICE_CPU)[i];
    leafToRegionMap[elt[0]].push_back(i);
  }

  // Make greyscale and colour images showing which pixels are in which leaves (for the first tree).
  cv::Mat1b imageG = cv::Mat1b::zeros(m_leafIndicesImage->noDims.y, m_leafIndicesImage->noDims.x);
  const uint32_t featureStep = m_featureCalculator->get_feature_step();
  for(std::map<int,std::vector<int> >::const_iterator jt = leafToRegionMap.begin(), jend = leafToRegionMap.end(); jt != jend; ++jt)
  {
    for(std::vector<int>::const_iterator kt = jt->second.begin(), kend = jt->second.end(); kt != kend; ++kt)
    {
      int x = *kt % m_leafIndicesImage->noDims.x, y = *kt / m_leafIndicesImage->noDims.x;
      if(depthImage->GetData(MEMORYDEVICE_CPU)[y * featureStep * depthImage->noDims.x + x * featureStep] > 0.0f)
      {
        imageG(y,x) = jt->first % 256;
      }
    }
  }

  cv::Mat3b imageC;
  cv::applyColorMap(imageG, imageC, cv::COLORMAP_HSV);

  // If the pixels to leaves image hasn't been allocated yet, allocate it now.
  if(!m_pixelsToLeavesImage) m_pixelsToLeavesImage.reset(new ORUChar4Image(Vector2i(imageC.cols, imageC.rows), true, true));

  for(int y = 0; y < imageC.rows; ++y)
  {
    for(int x = 0; x < imageC.cols; ++x)
    {
      const int offset = y * imageC.cols + x;
      cv::Vec3b& p = imageC(y, x);

      if(depthImage->GetData(MEMORYDEVICE_CPU)[y * featureStep * depthImage->noDims.x + x * featureStep] <= 0.0f)
      {
        p = cv::Vec3b(0,0,0);
      }

      m_pixelsToLeavesImage->GetData(MEMORYDEVICE_CPU)[offset] = Vector4u(p[2], p[1], p[0], 255);
    }
  }

  // Update the GPU copy of the pixels to leaves image (if it exists).
  m_pixelsToLeavesImage->UpdateDeviceFromHost();
#endif
}

void ScoreRelocaliser::update_pixels_to_points_image(const ORUtils::SE3Pose& worldToCamera) const
{
  // Ensure that the keypoints and SCoRe predictions are available on the CPU.
  m_keypointsImage->UpdateHostFromDevice();
  m_predictionsImage->UpdateHostFromDevice();

  // If the pixels to points image hasn't been allocated yet, allocate it now.
  if(!m_pixelsToPointsImage) m_pixelsToPointsImage.reset(new ORUChar4Image(m_leafIndicesImage->noDims, true, true));

  // For each pixel:
  Vector4u *p = m_pixelsToPointsImage->GetData(MEMORYDEVICE_CPU);
  for(int i = 0, pixelCount = static_cast<int>(m_pixelsToPointsImage->dataSize); i < pixelCount; ++i, ++p)
  {
    p->r = p->g = p->b = 0;
    p->a = 255;

    // If the pixel has a valid keypoint, look up the position of the cluster (if any) in the corresponding prediction that is closest to it.
    const ExampleType& keypoint = m_keypointsImage->GetData(MEMORYDEVICE_CPU)[i];
    if(!keypoint.valid) continue;
    const PredictionType& prediction = m_predictionsImage->GetData(MEMORYDEVICE_CPU)[i];
    const int closestModeIdx = find_closest_mode(worldToCamera.GetInvM() * keypoint.position, prediction);
    if(closestModeIdx == -1) continue;
    const Vector3f& clusterPos = prediction.elts[closestModeIdx].position;

    // Colour the pixel in the pixels to points image based on the cluster's position in world space.
    float scale = 2.0f;
    float centre = 0.5f;
    float offset = centre - 1.0f / (scale * 2.0f);
    float xCoeff = ((CLAMP(clusterPos.x,m_minX,m_maxX) - m_minX) / (m_maxX - m_minX)) / scale + offset;
    float yCoeff = ((CLAMP(clusterPos.y,m_minY,m_maxY) - m_minY) / (m_maxY - m_minY)) / scale + offset;
    float zCoeff = ((CLAMP(clusterPos.z,m_minZ,m_maxZ) - m_minZ) / (m_maxZ - m_minZ)) / scale + offset;
    p->r = static_cast<unsigned char>(255 * xCoeff);
    p->g = static_cast<unsigned char>(255 * yCoeff);
    p->b = static_cast<unsigned char>(255 * zCoeff);
    p->a = 255;
  }

  // Update the GPU copy of the pixels to points image (if it exists).
  m_pixelsToPointsImage->UpdateDeviceFromHost();
}

void ScoreRelocaliser::update_reservoir_start_idx()
{
  m_relocaliserState->reservoirUpdateStartIdx += m_maxReservoirsToUpdate;

  // If we go past the end of the list of reservoirs, loop back round.
  if(m_relocaliserState->reservoirUpdateStartIdx >= m_reservoirCount)
  {
    m_relocaliserState->reservoirUpdateStartIdx = 0;
  }
}

}
