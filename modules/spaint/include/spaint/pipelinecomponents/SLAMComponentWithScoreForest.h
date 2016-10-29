/**
 * spaint: SLAMComponentWithScoreForest.h
 * Copyright (c) Torr Vision Group, University of Oxford, 2016. All rights reserved.
 */

#ifndef H_SPAINT_SLAMCOMPONENTWITHSCOREFOREST
#define H_SPAINT_SLAMCOMPONENTWITHSCOREFOREST

#include "SLAMComponent.h"

#include <tuple>
#include <random>

#include <boost/optional.hpp>
#include <boost/shared_ptr.hpp>

#include <DatasetRGBD7Scenes.hpp>
#include <DFBP.hpp>

#include "../features/FeatureCalculatorFactory.h"
#include "../randomforest/interface/GPUForest.h"
#include "../randomforest/interface/GPURansac.h"

#include "tvgutil/filesystem/SequentialPathGenerator.h"

namespace spaint
{

/**
 * \brief An instance of this pipeline component can be used to perform simultaneous localisation and mapping (SLAM).
 */
class SLAMComponentWithScoreForest: public SLAMComponent
{
//  struct PoseCandidate
//  {
//    struct Inlier
//    {
//      int linearIdx;
//      int modeIdx;
//      float energy;
//    };
//
//    Matrix4f cameraPose;
//    std::vector<Inlier> inliers;
//    float energy;
//    int cameraId;
//  };

  //#################### CONSTRUCTORS ####################
public:
  /**
   * \brief Constructs a SLAM component performing relocalization using a Score Forest.
   *
   * \param context           The shared context needed for SLAM.
   * \param sceneID           The ID of the scene to reconstruct.
   * \param imageSourceEngine The engine used to provide input images to the fusion process.
   * \param trackerType       The type of tracker to use.
   * \param trackerParams     The parameters for the tracker (if any).
   * \param mappingMode       The mapping mode to use.
   * \param trackingMode      The tracking mode to use.
   */
  SLAMComponentWithScoreForest(const SLAMContext_Ptr& context,
      const std::string& sceneID,
      const ImageSourceEngine_Ptr& imageSourceEngine, TrackerType trackerType,
      const std::vector<std::string>& trackerParams, MappingMode mappingMode =
          MAP_VOXELS_ONLY, TrackingMode trackingMode = TRACK_VOXELS);

  //#################### DESTRUCTOR ####################
public:
  /**
   * \brief Destroys a SLAM component.
   */
  virtual ~SLAMComponentWithScoreForest();

  //#################### PROTECTED MEMBER FUNCTIONS ####################
protected:
  virtual TrackingResult process_relocalisation(TrackingResult trackingResult);

  //#################### PRIVATE MEMBER FUNCTIONS ####################
private:
  boost::optional<PoseCandidate> estimate_pose();
  void compute_features(const ITMUChar4Image_CPtr &inputRgbImage,
      const ITMFloatImage_CPtr &inputDepthImage,
      const Vector4f &depthIntrinsics, const Matrix4f &invCameraPose);
  void compute_features(const ITMUChar4Image_CPtr &inputRgbImage,
      const ITMFloatImage_CPtr &inputDepthImage,
      const Vector4f &depthIntrinsics);
  void evaluate_forest();

//  void generate_pose_candidates(std::vector<PoseCandidate> &poseCandidates);
//  bool hypothesize_pose(PoseCandidate &res, std::mt19937 &eng);
//  void sample_pixels_for_ransac(std::vector<bool> &maskSampledPixels,
//      std::vector<Vector2i> &sampledPixelIdx, std::mt19937 &eng, int batchSize);
//  void update_inliers_for_optimization(
//      const std::vector<Vector2i> &sampledPixelIdx,
//      std::vector<PoseCandidate> &poseCandidates) const;
//  void compute_and_sort_energies(
//      std::vector<PoseCandidate> &poseCandidates) const;
//  float compute_pose_energy(const Matrix4f &candidateCameraPose,
//      std::vector<PoseCandidate::Inlier> &inliers) const;
//  void update_candidate_poses(std::vector<PoseCandidate> &poseCandidates) const;
//  bool update_candidate_pose(PoseCandidate &poseCandidate) const;

  //#################### PRIVATE MEMBER VARIABLES ####################
private:
  boost::shared_ptr<DatasetRGBD7Scenes> m_dataset;
  boost::shared_ptr<DFBP> m_forest;
  RGBDPatchFeatureCalculator_CPtr m_featureExtractor;
  RGBDPatchFeatureImage_Ptr m_featureImage;
  GPUForestPredictionsImage_Ptr m_predictionsImage;
  GPUForest_Ptr m_gpuForest;
  GPURansac_Ptr m_gpuRansac;

  Tracker_Ptr m_refineTracker;
  boost::optional<tvgutil::SequentialPathGenerator> m_sequentialPathGenerator;

  // Member variables from scoreforests
  size_t m_kInitRansac;
  size_t m_nbPointsForKabschBoostrap;
  bool m_useAllModesPerLeafInPoseHypothesisGeneration;
  bool m_checkMinDistanceBetweenSampledModes;
  float m_minDistanceBetweenSampledModes;
  bool m_checkRigidTransformationConstraint;
  float m_translationErrorMaxForCorrectPose;
  size_t m_batchSizeRansac;
  size_t m_trimKinitAfterFirstEnergyComputation;
  bool m_poseUpdate;
  bool m_usePredictionCovarianceForPoseOptimization;
  float m_poseOptimizationInlierThreshold;

  // Additional parameters for online evaluation
  int m_maxNbModesPerLeaf;
};

//#################### TYPEDEFS ####################

typedef boost::shared_ptr<SLAMComponentWithScoreForest> SLAMComponentWithScoreForest_Ptr;

}

#endif
