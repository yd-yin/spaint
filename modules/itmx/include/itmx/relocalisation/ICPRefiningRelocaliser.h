/**
 * itmx: ICPRefiningRelocaliser.h
 * Copyright (c) Torr Vision Group, University of Oxford, 2017. All rights reserved.
 */

#ifndef H_ITMX_ICPREFININGRELOCALISER
#define H_ITMX_ICPREFININGRELOCALISER

#include <boost/optional.hpp>

#ifdef WITH_OPENCV
#include <opencv2/core/core.hpp>
#endif

#include <ITMLib/Core/ITMDenseMapper.h>
#include <ITMLib/Engines/Visualisation/Interface/ITMVisualisationEngine.h>
#include <ITMLib/Objects/Scene/ITMScene.h>

#include <orx/relocalisation/RefiningRelocaliser.h>

#include <tvgutil/filesystem/SequentialPathGenerator.h>

#include "../base/ITMObjectPtrTypes.h"
#include "../visualisation/interface/DepthVisualiser.h"

namespace itmx {

/**
 * \brief An instance of this class can be used to refine the results of another relocaliser using ICP.
 *
 * \tparam VoxelType  The type of voxel used to reconstruct the scene that will be used during the raycasting step.
 * \tparam IndexType  The type of indexing used to access the reconstructed scene.
 */
template <typename VoxelType, typename IndexType>
class ICPRefiningRelocaliser : public orx::RefiningRelocaliser
{
  //#################### TYPEDEFS ####################
private:
  typedef ITMLib::ITMDenseMapper<VoxelType,IndexType> DenseMapper;
  typedef boost::shared_ptr<DenseMapper> DenseMapper_Ptr;
  typedef ITMLib::ITMScene<VoxelType,IndexType> Scene;
  typedef boost::shared_ptr<Scene> Scene_Ptr;
  typedef ITMLib::ITMVisualisationEngine<VoxelType,IndexType> VisualisationEngine;
  typedef boost::shared_ptr<const VisualisationEngine> VisualisationEngine_CPtr;

  //#################### PRIVATE MEMBER VARIABLES ####################
private:
  /** Whether or not to choose the best result. */
  bool m_chooseBestResult;

  /** The dense mapper used to find visible blocks in the voxel scene. */
  DenseMapper_Ptr m_denseVoxelMapper;

  /** The depth visualiser. */
  DepthVisualiser_CPtr m_depthVisualiser;

  /** The path generator used to find the ground truth pose files. */
  mutable boost::optional<tvgutil::SequentialPathGenerator> m_gtPathGenerator;

  /** The path generator used when saving the relocalised images. */
  mutable boost::optional<tvgutil::SequentialPathGenerator> m_imagePathGenerator;

  /** The path generator used when saving the relocalised poses. */
  mutable boost::optional<tvgutil::SequentialPathGenerator> m_posePathGenerator;

  /** Whether or not to save the images rendered from the relocalised poses. */
  bool m_saveImages;

  /** Whether or not to save the relocalised poses. */
  bool m_savePoses;

  /** Whether or not to save the average relocalisation times. */
  bool m_saveTimes;

  /** The scene being viewed from the camera. */
  Scene_Ptr m_scene;

  /** The settings to use for InfiniTAM. */
  Settings_CPtr m_settings;

  /** The timer used to profile the initial relocalisations. */
  mutable AverageTimer m_timerInitialRelocalisation;

  /** The timer used to profile the ICP refinement. */
  mutable AverageTimer m_timerRefinement;

  /** The timer used to profile the relocalisation calls. */
  mutable AverageTimer m_timerRelocalisation;

  /** The path to a file in which to save the average relocalisation times. */
  std::string m_timersOutputFile;

  /** The timer used to profile the training calls. */
  AverageTimer m_timerTraining;

  /** The timer used to profile the update calls. */
  AverageTimer m_timerUpdate;

  /** The ICP tracker used to refine the relocalised poses. */
  Tracker_Ptr m_tracker;

  /** The tracking controller used to set up and perform the actual refinement. */
  TrackingController_Ptr m_trackingController;

  /** The tracking state used to hold the refinement results. */
  TrackingState_Ptr m_trackingState;

  /** The visualisation engine used to perform the raycasting. */
  VisualisationEngine_CPtr m_visualisationEngine;

  /** The current view of the scene. */
  View_Ptr m_view;

  /** The voxel render state used to hold the raycasting results. */
  mutable VoxelRenderState_Ptr m_voxelRenderState;

  //#################### CONSTRUCTORS ####################
public:
  /**
   * \brief Constructs an ICP-based refining relocaliser.
   *
   * \param innerRelocaliser    The relocaliser whose results are being refined using ICP.
   * \param tracker             The ICP tracker.
   * \param rgbImageSize        The size of the colour images produced by the camera.
   * \param depthImageSize      The size of the depth images produced by the camera.
   * \param calib               The calibration parameters of the camera whose pose is to be estimated.
   * \param scene               The scene being viewed from the camera.
   * \param denseVoxelMapper    The dense mapper used to find visible blocks in the voxel scene.
   * \param settings            The settings to use for InfiniTAM.
   */
  ICPRefiningRelocaliser(const orx::Relocaliser_Ptr& innerRelocaliser, const Tracker_Ptr& tracker,
                         const Vector2i& rgbImageSize, const Vector2i& depthImageSize,
                         const ITMLib::ITMRGBDCalib& calib, const Scene_Ptr& scene,
                         const DenseMapper_Ptr& denseVoxelMapper, const Settings_CPtr& settings);

  //#################### DESTRUCTOR ####################
public:
  /**
   * \brief Destroys the relocaliser.
   */
  ~ICPRefiningRelocaliser();

  //#################### COPY CONSTRUCTOR & ASSIGNMENT OPERATOR ####################
private:
  // Deliberately private and unimplemented.
  ICPRefiningRelocaliser(const ICPRefiningRelocaliser&);
  ICPRefiningRelocaliser& operator=(const ICPRefiningRelocaliser&);

  //#################### PUBLIC MEMBER FUNCTIONS ####################
public:
  /** Override */
  virtual void finish_training();

  /** Override */
  virtual ORUChar4Image_CPtr get_visualisation_image(const std::string& key) const;

  /** Override */
  virtual void load_from_disk(const std::string& inputFolder);

  /** Override */
  virtual std::vector<Result> relocalise(const ORUChar4Image *colourImage, const ORFloatImage *depthImage,
                                         const Vector4f& depthIntrinsics) const;

  /** Override */
  virtual std::vector<Result> relocalise(const ORUChar4Image *colourImage, const ORFloatImage *depthImage,
                                         const Vector4f& depthIntrinsics, std::vector<ORUtils::SE3Pose>& initialPoses) const;

  /** Override */
  virtual void reset();

  /** Override */
  virtual void save_to_disk(const std::string& outputFolder) const;

  /** Override */
  virtual void train(const ORUChar4Image *colourImage, const ORFloatImage *depthImage,
                     const Vector4f& depthIntrinsics, const ORUtils::SE3Pose& cameraPose);

  /** Override */
  virtual void update();

  //#################### PRIVATE MEMBER FUNCTIONS ####################
private:
#ifdef WITH_OPENCV
  /**
   * \brief Computes a difference image between two depth images, and saves it to disk.
   *
   * \param depthImage1 The first depth image.
   * \param depthImage2 The second depth image.
   * \param pattern     The pattern to use when constructing the name of the file into which to save the difference image.
   */
  void compute_and_save_diff(const cv::Mat& depthImage1, const cv::Mat& depthImage2, const std::string& pattern) const;
#endif

#ifdef WITH_OPENCV
  /**
   * \brief Makes a colourised version of a floating-point depth image and saves it to disk.
   *
   * \param depthF  The floating-point depth image to colourise and save.
   * \param depthU  A temporary image in which to store the colourised version of the depth.
   * \param pattern The pattern to use when constructing the name of the file into which to save the image.
   */
  void save_colourised_depth(const ORFloatImage *depthF, const ORUChar4Image_Ptr& depthU, const std::string& pattern) const;
#endif

  /**
   * \brief Saves the relocalised and refined poses in text files so that they can be used later (e.g. for evaluation).
   *
   * \note Saving happens only if m_savePoses is true.
   *
   * \param relocalisedPose The relocalised pose before refinement.
   * \param refinedPose     The result of refining the relocalised pose.
   */
  void save_poses(const Matrix4f& relocalisedPose, const Matrix4f& refinedPose) const;

  /**
   * \brief Scores a proposed camera pose by computing the mean depth difference between the real depth image
   *        and a synthetic depth image rendered from it.
   *
   * \param pose  The pose to score.
   * \return      The score computed for the pose.
   */
  float score_pose(const ORUtils::SE3Pose& pose) const;
};

}

#endif
