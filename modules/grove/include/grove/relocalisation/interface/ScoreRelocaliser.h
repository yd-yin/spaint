/**
 * grove: ScoreRelocaliser.h
 * Copyright (c) Torr Vision Group, University of Oxford, 2017. All rights reserved.
 */

#ifndef H_GROVE_SCORERELOCALISER
#define H_GROVE_SCORERELOCALISER

#include <boost/optional.hpp>
#include <boost/thread.hpp>

#include <ORUtils/DeviceType.h>

#include <orx/relocalisation/Relocaliser.h>

#include "../base/ScoreRelocaliserState.h"
#include "../../clustering/interface/ExampleClusterer.h"
#include "../../features/interface/RGBDPatchFeatureCalculator.h"
#include "../../forests/interface/DecisionForest.h"
#include "../../ransac/interface/PreemptiveRansac.h"
#include "../../reservoirs/interface/ExampleReservoirs.h"
#include "../../scoreforests/Keypoint3DColourCluster.h"
#include "../../scoreforests/ScorePrediction.h"

namespace grove {

/**
 * \brief An instance of a class deriving from this one can be used to relocalise a camera in a 3D scene, using the approach described
 *        in "On-the-Fly Adaptation of Regression Forests for Online Camera Relocalisation" (Cavallari et al., 2017).
 */
class ScoreRelocaliser : public orx::Relocaliser
{
  //#################### CONSTANTS ####################
public:
  enum { FOREST_TREE_COUNT = 5 };

  //#################### TYPEDEFS ####################
public:
  typedef Keypoint3DColour ExampleType;
  typedef Keypoint3DColourCluster ClusterType;
  typedef RGBDPatchDescriptor DescriptorType;
  typedef ScorePrediction PredictionType;

  typedef ExampleClusterer<ExampleType, ClusterType, PredictionType::Capacity> Clusterer;
  typedef boost::shared_ptr<Clusterer> Clusterer_Ptr;

  typedef ORUtils::VectorX<int, FOREST_TREE_COUNT> LeafIndices;
  typedef ORUtils::Image<LeafIndices> LeafIndicesImage;
  typedef boost::shared_ptr<LeafIndicesImage> LeafIndicesImage_Ptr;
  typedef boost::shared_ptr<const LeafIndicesImage> LeafIndicesImage_CPtr;

  typedef ExampleReservoirs<ExampleType> Reservoirs;
  typedef boost::shared_ptr<Reservoirs> Reservoirs_Ptr;

  typedef DecisionForest<DescriptorType, FOREST_TREE_COUNT> ScoreForest;
  typedef boost::shared_ptr<ScoreForest> ScoreForest_Ptr;

//#################### PRIVATE VARIABLES ####################
private:
  /** The image containing the descriptors extracted from the RGB-D image. */
  RGBDPatchDescriptorImage_Ptr m_descriptorsImage;

  /** The image containing the keypoints extracted from the RGB-D image. */
  Keypoint3DColourImage_Ptr m_keypointsImage;

  /** The image containing the indices of the forest leaves associated with the keypoint/descriptor pairs. */
  mutable LeafIndicesImage_Ptr m_leafIndicesImage;

  /** The mutex used to synchronise access to the relocaliser in a multithreaded environment. */
  mutable boost::recursive_mutex m_mutex;

  /** The image containing the forest predictions associated with the keypoint/descriptor pairs. */
  mutable ScorePredictionsImage_Ptr m_predictionsImage;

  //#################### PROTECTED VARIABLES ####################
protected:
  /** A flag indicating whether or not this relocaliser is "backed" by another one. */
  bool m_backed;

  /** The sigma of the Gaussian used when computing the example densities (used during clustering). */
  float m_clustererSigma;

  /** The maximum distance there can be between two examples that are part of the same cluster (used during clustering). */
  float m_clustererTau;

  /** The device on which the relocaliser should operate. */
  ORUtils::DeviceType m_deviceType;

  /** The clusterer used to compute 3D modal clusters from the examples stored in the reservoirs. */
  Clusterer_Ptr m_exampleClusterer;

  /** The feature calculator used to extract keypoints and descriptors from the RGB-D image. */
  DA_RGBDPatchFeatureCalculator_Ptr m_featureCalculator;

  /** The maximum number of clusters to store in each leaf in the forest (used during clustering). */
  uint32_t m_maxClusterCount;

  /** The maximum number of relocalisations to output for each call to the relocalise function. */
  uint32_t m_maxRelocalisationsToOutput;

  /** The maximum number of reservoirs to subject to clustering for each call to the update function. */
  uint32_t m_maxReservoirsToUpdate;

  /** The maximum x, y and z coordinates visited by the camera during training. */
  float m_maxX, m_maxY, m_maxZ;

  /** The minimum size of cluster to keep (used during clustering). */
  uint32_t m_minClusterSize;

  /** The minimum x, y and z coordinates visited by the camera during training. */
  float m_minX, m_minY, m_minZ;

  /** An image in which to store a visualisation of the mapping from pixels to forest leaves (for debugging purposes). */
  mutable ORUChar4Image_Ptr m_pixelsToLeavesImage;

  /** An image in which to store a visualisation of the mapping from pixels to world-space points (for debugging purposes). */
  mutable ORUChar4Image_Ptr m_pixelsToPointsImage;

  /** The Preemptive RANSAC instance, used to estimate the 6DOF camera pose from a set of 3D keypoints and their associated SCoRe forest predictions. */
  PreemptiveRansac_Ptr m_preemptiveRansac;

  /** The state of the relocaliser. Can be replaced at runtime to relocalise (and train) in a different environment. */
  ScoreRelocaliserState_Ptr m_relocaliserState;

  /** The capacity (maximum size) of each reservoir associated with a leaf in the forest. */
  uint32_t m_reservoirCapacity;

  /** The total number of example reservoirs used by the relocaliser (in practice, this is equal to the number of leaves in the forest). */
  uint32_t m_reservoirCount;

  /** The seed for the random number generators used by the example reservoirs. */
  uint32_t m_rngSeed;

  /** The SCoRe forest on which the relocaliser is based. */
  ScoreForest_Ptr m_scoreForest;

  /** The settings used to configure the relocaliser. */
  tvgutil::SettingsContainer_CPtr m_settings;

  /** Whether or not to produce visualisations of the forest when relocalising. */
  bool m_visualiseForest;

  //#################### CONSTRUCTORS ####################
protected:
  /**
   * \brief Constructs a SCoRe relocaliser by loading a pre-trained forest from a file.
   *
   * \param forestFilename    The name of the file from which to load the pre-trained forest.
   * \param settings          The settings used to configure the relocaliser.
   * \param settingsNamespace The namespace associated with the settings that are specific to the SCoRe relocaliser.
   * \param deviceType        The device on which the relocaliser should operate.
   *
   * \throws std::runtime_error If the forest cannot be loaded.
   */
  ScoreRelocaliser(const std::string& forestFilename, const tvgutil::SettingsContainer_CPtr& settings, const std::string& settingsNamespace, ORUtils::DeviceType deviceType);

  //#################### DESTRUCTOR ####################
public:
  /**
   * \brief Destroys the relocaliser.
   */
  virtual ~ScoreRelocaliser();

  //#################### PROTECTED ABSTRACT MEMBER FUNCTIONS ####################
protected:
  /**
   * \brief Counts the number of pixels in the specified depth image that contain a valid depth value.
   *
   * \param depthImage  The depth image.
   *
   * \return  The number of pixels in the depth image that contain a valid depth value.
   */
  virtual uint32_t count_valid_depths(const ORFloatImage *depthImage) const = 0;

  /**
   * \brief Merges the SCoRe predictions (sets of clusters) associated with each keypoint to create a single
   *        SCoRe prediction (a single set of clusters) for each keypoint.
   *
   * \note  Each keypoint/descriptor pair extracted from the input RGB-D image pairs determines a leaf in a tree of the
   *        forest. Each such leaf contains a set of 3D modal clusters, which together constitute a SCoRe prediction.
   *        This function merges the SCoRe predictions associated with the different leaves (from different trees) with
   *        which each keypoint/descriptor pair is associated, thereby yielding a single SCoRe prediction for each pair.
   *
   * \param leafIndices       An image containing the indices of the leaves (in the different trees) associated with each keypoint/descriptor pair.
   * \param outputPredictions An image into which to store the merged SCoRe predictions.
   */
  virtual void merge_predictions_for_keypoints(const LeafIndicesImage_CPtr& leafIndices, ScorePredictionsImage_Ptr& outputPredictions) const = 0;

  //#################### PUBLIC MEMBER FUNCTIONS ####################
public:
  /** Override */
  virtual void finish_training();

  /**
   * \brief Gets all of the candidate poses that survived the initial culling process during the last run of P-RANSAC,
   *        sorted in non-increasing order of the number of P-RANSAC iterations they survived.
   *
   * \pre   This function should only be called after a prior call to relocalise.
   * \note  The first entry of the vector will be the candidate (if any) returned by the last run of P-RANSAC.
   *
   * \param poseCandidates An output array that will be filled with the candidate poses as described.
   */
  void get_best_poses(std::vector<PoseCandidate>& poseCandidates) const;

  /**
   * \brief Gets the image containing the keypoints extracted from the RGB-D image.
   *
   * \return  The image containing the keypoints extracted from the RGB-D image.
   */
  Keypoint3DColourImage_CPtr get_keypoints_image() const;

  /**
   * \brief Gets the prediction associated with the specified leaf in the forest.
   *
   * \param treeIdx The index of the tree containing the prediction.
   * \param leafIdx The index of the leaf.
   * \return        The prediction associated with the specified leaf.
   *
   * \throws std::invalid_argument  If treeIdx or leafIdx are greater than the maximum number of trees or leaves, respectively.
   */
  ScorePrediction get_prediction(uint32_t treeIdx, uint32_t leafIdx) const;

  /**
   * \brief Gets the image containing the forest predictions associated with the keypoint/descriptor pairs.
   *
   * \return  The image containing the forest predictions associated with the keypoint/descriptor pairs.
   */
  ScorePredictionsImage_CPtr get_predictions_image() const;

  /**
   * \brief Gets the contents of the reservoir associated with the specified leaf in the forest.
   *
   * \param treeIdx The index of the tree containing the leaf.
   * \param leafIdx The index of the leaf.
   * \return        The reservoir associated with the specified leaf.
   *
   * \throws std::invalid_argument  If treeIdx or leafIdx are greater than the maximum number of trees or leaves, respectively.
   */
  std::vector<Keypoint3DColour> get_reservoir_contents(uint32_t treeIdx, uint32_t leafIdx) const;

  /** Override */
  virtual ORUChar4Image_CPtr get_visualisation_image(const std::string& key) const;

  /** Override */
  virtual void load_from_disk(const std::string& inputFolder);

  /** Override */
  virtual std::vector<Result> relocalise(const ORUChar4Image *colourImage, const ORFloatImage *depthImage, const Vector4f& depthIntrinsics) const;
  std::vector<Result> relocalise_rel(const ORUChar4Image *colourImage, const ORFloatImage *depthImage, const Vector4f& depthIntrinsics, int fid);

  /** Override */
  virtual void reset();

  /** Override */
  virtual void save_to_disk(const std::string& outputFolder) const;

  /**
   * \brief Replaces the relocaliser's current state with that of another relocaliser, and marks this relocaliser as being "backed" by that relocaliser.
   *
   * \note  The new state must previously have been initialised with the right variable sizes.
   *
   * \param backingRelocaliser  The backing relocaliser.
   */
  void set_backing_relocaliser(const boost::shared_ptr<ScoreRelocaliser>& backingRelocaliser);

  /** Override */
  virtual void train(const ORUChar4Image *colourImage, const ORFloatImage *depthImage, const Vector4f& depthIntrinsics, const ORUtils::SE3Pose& cameraPose);

  /** Override */
  virtual void test4pcd(const ORUChar4Image *colourImage, const ORFloatImage *depthImage, const Vector4f& depthIntrinsics, const ORUtils::SE3Pose& cameraPose, CorrectPointCloud& pointCloud);

  /** Override */
  virtual void update();

  /**
   * \brief Forcibly updates the contents of every cluster in the forest.
   *
   * \note  This is computationally intensive, and can require a few hundred milliseconds to terminate.
   */
  void update_all_clusters();

  //#################### PRIVATE MEMBER FUNCTIONS ####################
private:
  /**
   * \brief Computes the number of reservoirs to subject to clustering during a train/update call.
   *
   * \return  The number of reservoirs to subject to clustering during a train/update call.
   */
  uint32_t compute_nb_reservoirs_to_update() const;

  /**
   * \brief Checks whether or not the specified leaf is valid, and throws if not.
   *
   * \param treeIdx The index of the tree containing the leaf.
   * \param leafIdx The index of the leaf.
   *
   * \throws std::invalid_argument  If treeIdx or leafIdx are greater than the maximum number of trees or leaves, respectively.
   */
  void ensure_valid_leaf(uint32_t treeIdx, uint32_t leafIdx) const;

  /**
   * \brief Updates the pixels to leaves image (for debugging purposes).
   *
   * \param depthImage  The current depth image.
   */
  void update_pixels_to_leaves_image(const ORFloatImage *depthImage) const;

  /**
   * \brief Updates the pixels to points image (for debugging purposes).
   *
   * \param worldToCamera The relocalised pose.
   */
  void update_pixels_to_points_image(const ORUtils::SE3Pose& worldToCamera) const;

  /**
   * \brief Updates the index of the first reservoir to subject to clustering during the next train/update call.
   */
  void update_reservoir_start_idx();
};

//#################### TYPEDEFS ####################

typedef boost::shared_ptr<ScoreRelocaliser> ScoreRelocaliser_Ptr;
typedef boost::shared_ptr<const ScoreRelocaliser> ScoreRelocaliser_CPtr;

}

#endif
