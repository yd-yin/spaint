/**
 * itmx: ICPRefiningRelocaliser.h
 * Copyright (c) Torr Vision Group, University of Oxford, 2017. All rights reserved.
 */

#ifndef H_ITMX_ICPREFININGRELOCALISER
#define H_ITMX_ICPREFININGRELOCALISER

#include <boost/shared_ptr.hpp>

#include <ITMLib/Core/ITMDenseMapper.h>
#include <ITMLib/Engines/Visualisation/Interface/ITMVisualisationEngine.h>
#include <ITMLib/Objects/Scene/ITMScene.h>

#include <tvgutil/filesystem/SequentialPathGenerator.h>
#include <tvgutil/timing/AverageTimer.h>

#include "../base/ITMObjectPtrTypes.h"
#include "RefiningRelocaliser.h"

namespace itmx {

/**
 * \brief An instance of this class can be used to refine the results of another relocaliser using ICP.
 *
 * \tparam VoxelType  The type of voxel used to recontruct the scene that will be used during the raycasting step.
 * \tparam IndexType  The type of indexing used to access the reconstructed scene.
 */
template <typename VoxelType, typename IndexType>
class ICPRefiningRelocaliser : public RefiningRelocaliser
{
  //#################### TYPEDEFS ####################
public:
  typedef ITMLib::ITMDenseMapper<VoxelType,IndexType> DenseMapper;
  typedef boost::shared_ptr<DenseMapper> DenseMapper_Ptr;

  typedef ITMLib::ITMScene<VoxelType,IndexType> Scene;
  typedef boost::shared_ptr<Scene> Scene_Ptr;

  typedef ITMLib::ITMVisualisationEngine<VoxelType,IndexType> VisualisationEngine;
  typedef boost::shared_ptr<VisualisationEngine> VisualisationEngine_Ptr;

  //#################### TYPEDEFS ####################
private:
  typedef tvgutil::AverageTimer<boost::chrono::microseconds> AverageTimer;

  //#################### CONSTRUCTORS ####################
public:
  /**
   * \brief Constructs an ICP-based refining relocaliser.
   *
   * \param relocaliser     The relocaliser whose results will be refined using ICP.
   * \param calib           The calibration parameters of the camera whose pose is to be estimated.
   * \param rgbImageSize    The size of the colour images produced by the camera.
   * \param depthImageSize  The size of the depth images produced by the camera.
   * \param scene           The scene being viewed from the camera.
   * \param settings        The settings to use for InfiniTAM.
   * \param trackerConfig   A configuration string used to specify the parameters of the ICP tracker.
   */
  ICPRefiningRelocaliser(const Relocaliser_Ptr& relocaliser, const ITMLib::ITMRGBDCalib& calib,
                         const Vector2i& rgbImageSize, const Vector2i& depthImageSize, const Scene_Ptr& scene,
                         const Settings_CPtr& settings, const std::string& trackerConfig);

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
  virtual boost::optional<Result> relocalise(const ITMUChar4Image *colourImage, const ITMFloatImage *depthImage,
                                             const Vector4f &depthIntrinsics) const;

  /** Override */
  virtual boost::optional<Result> relocalise(const ITMUChar4Image *colourImage, const ITMFloatImage *depthImage,
                                             const Vector4f &depthIntrinsics, boost::optional<ORUtils::SE3Pose> &initialPose) const;

  /** Override */
  virtual void reset();

  /** Override */
  virtual void train(const ITMUChar4Image *colourImage, const ITMFloatImage *depthImage,
                     const Vector4f& depthIntrinsics, const ORUtils::SE3Pose& cameraPose);

  /** Override */
  virtual void update();

  //#################### PRIVATE MEMBER FUNCTIONS ####################
private:
  /**
   * \brief This function saves the relocalised and refined poses in text files used for evaluation.
   *
   * \note Saving happens only if m_saveRelocalisationPoses is true.
   *
   * \param relocalisedPose     The relocalised pose.
   * \param refinedPose         The pose after refinement.
   */
  void save_poses(const Matrix4f &relocalisedPose, const Matrix4f &refinedPose) const;

  /**
   * \brief Starts a timer (waiting for all CUDA operations to terminate first, if necessary).
   *
   * \param timer The timer.
   */
  void start_timer(AverageTimer &timer) const;

  /**
   * \brief Stops a timer (waiting for all CUDA operations to terminate first, if necessary).
   *
   * \param timer The timer.
   */
  void stop_timer(AverageTimer &timer) const;

  //#################### PRIVATE MEMBER VARIABLES ####################
private:
  /** A DenseMapper used to find visible blocks in the scene. */
  DenseMapper_Ptr m_denseMapper;

  /** A low level engine used by the tracker. */
  LowLevelEngine_Ptr m_lowLevelEngine;

  /** The path generator used when saving the relocalised poses. */
  mutable boost::optional<tvgutil::SequentialPathGenerator> m_relocalisationPosesPathGenerator;

  /** Whether or not to save the relocalised poses. */
  bool m_saveRelocalisationPoses;

  /** The reconstructed scene. */
  Scene_Ptr m_scene;

  /** Settings used when reconstructing the scene. */
  Settings_CPtr m_settings;

  /** The timer used to profile the integration calls. */
  AverageTimer m_timerIntegration;

  /** The timer used to profile the relocalisation calls. */
  mutable AverageTimer m_timerRelocalisation;

  /** The timer used to profile the update calls. */
  AverageTimer m_timerUpdate;

  /** Whether or not timers are enabled and stats are printed on destruction. */
  bool m_timersEnabled;

  /** A tracker used to refine the relocalised poses. */
  Tracker_Ptr m_tracker;

  /** A tracking controller used to setup and perform the actual refinement. */
  TrackingController_Ptr m_trackingController;

  /** A tracking state used to hold refinement results. */
  TrackingState_Ptr m_trackingState;

  /** A visualization engine used to perform the raycasting. */
  VisualisationEngine_Ptr m_visualisationEngine;

  /** A view used to pass the input images to the tracker and the visualization engine. */
  View_Ptr m_view;

  /** A renderState used to hold the raycasting results. */
  mutable VoxelRenderState_Ptr m_voxelRenderState;
};

}

#endif
