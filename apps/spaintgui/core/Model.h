/**
 * spaintgui: Model.h
 * Copyright (c) Torr Vision Group, University of Oxford, 2015. All rights reserved.
 */

#ifndef H_SPAINTGUI_MODEL
#define H_SPAINTGUI_MODEL

#include <boost/shared_ptr.hpp>

#include <InputSource/ImageSourceEngine.h>

#include <ITMLib/Objects/Scene/ITMScene.h>
#include <ITMLib/Objects/Tracking/ITMTrackingState.h>
#include <ITMLib/Objects/Views/ITMView.h>
#include <ITMLib/Utils/ITMLibSettings.h>

#include <spaint/pipelinecomponents/SLAMModel.h>
#include <spaint/pipelinecomponents/SmoothingModel.h>
#include <spaint/util/LabelManager.h>

/**
 * \brief An instance of this class represents our model of the spaint scenario.
 *
 * The scenario we model is one of reconstructing a scene from a series of RGB-D images with known (tracked) pose,
 * and labelling it interactively using various user input modalities.
 */
class Model
: public spaint::SLAMModel,
  public spaint::SmoothingModel
{
  //#################### TYPEDEFS ####################
public:
  typedef ITMLib::ITMScene<spaint::SpaintVoxel,ITMVoxelIndex> Scene;
  typedef boost::shared_ptr<Scene> Scene_Ptr;
  typedef boost::shared_ptr<const Scene> Scene_CPtr;
  typedef boost::shared_ptr<const ITMLib::ITMLibSettings> Settings_CPtr;
  typedef boost::shared_ptr<ITMLib::ITMTrackingState> TrackingState_Ptr;
  typedef boost::shared_ptr<const ITMLib::ITMTrackingState> TrackingState_CPtr;
  typedef boost::shared_ptr<ITMLib::ITMView> View_Ptr;
  typedef boost::shared_ptr<const ITMLib::ITMView> View_CPtr;
  typedef ITMLib::ITMVisualisationEngine<spaint::SpaintVoxel,ITMVoxelIndex> VisualisationEngine;
  typedef boost::shared_ptr<VisualisationEngine> VisualisationEngine_Ptr;
  typedef boost::shared_ptr<const VisualisationEngine> VisualisationEngine_CPtr;

  //#################### PRIVATE VARIABLES ####################
private:
  /** The dimensions of the depth images from which the scene is being reconstructed. */
  Vector2i m_depthImageSize;

  /** The label manager. */
  spaint::LabelManager_Ptr m_labelManager;

  /** The path to the resources directory. */
  std::string m_resourcesDir;

  /** The dimensions of the RGB images from which the scene is being reconstructed. */
  Vector2i m_rgbImageSize;

  /** The current reconstructed scene. */
  Scene_Ptr m_scene;

  /** The settings to use for InfiniTAM. */
  Settings_CPtr m_settings;

  /** The current tracking state (containing the camera pose and additional tracking information used by InfiniTAM). */
  TrackingState_Ptr m_trackingState;

  /** The current view of the scene. */
  View_Ptr m_view;

  /** The InfiniTAM engine used for raycasting the scene. */
  VisualisationEngine_Ptr m_visualisationEngine;

  //#################### CONSTRUCTORS ####################
public:
  /**
   * \brief Constructs a model.
   *
   * \param scene           The InfiniTAM scene.
   * \param rgbImageSize    The dimensions of the RGB images from which the scene is being reconstructed.
   * \param depthImageSize  The dimensions of the depth images from which the scene is being reconstructed.
   * \param trackingState   The current tracking state (containing the camera pose and additional tracking information used by InfiniTAM).
   * \param settings        The settings to use for InfiniTAM.
   * \param resourcesDir    The path to the resources directory.
   * \param labelManager    The label manager.
   */
  Model(const Scene_Ptr& scene, const Vector2i& rgbImageSize, const Vector2i& depthImageSize, const TrackingState_Ptr& trackingState,
        const Settings_CPtr& settings, const std::string& resourcesDir, const spaint::LabelManager_Ptr& labelManager);

  //#################### PUBLIC MEMBER FUNCTIONS ####################
public:
  /**
   * \brief Gets the dimensions of the depth images from which the scene is being reconstructed.
   *
   * \return  The dimensions of the depth images from which the scene is being reconstructed.
   */
  const Vector2i& get_depth_image_size() const;

  /**
   * \brief Gets the intrinsic parameters for the camera that is being used to reconstruct the scene.
   *
   * \return  The intrinsic parameters for the camera.
   */
  const ITMLib::ITMIntrinsics& get_intrinsics() const;

  /**
   * \brief Gets the label manager.
   *
   * \return  The label manager.
   */
  const spaint::LabelManager_Ptr& get_label_manager();

  /**
   * \brief Gets the label manager.
   *
   * \return  The label manager.
   */
  spaint::LabelManager_CPtr get_label_manager() const;

  /**
   * \brief Gets the current pose of the camera that is being used to reconstruct the scene.
   *
   * \return  The current camera pose.
   */
  const ORUtils::SE3Pose& get_pose() const;

  /**
   * \brief Gets the path to the resources directory.
   *
   * \return The path to the resources directory.
   */
  const std::string& get_resources_dir() const;

  /**
   * \brief Gets the dimensions of the RGB images from which the scene is being reconstructed.
   *
   * \return  The dimensions of the RGB images from which the scene is being reconstructed.
   */
  const Vector2i& get_rgb_image_size() const;

  /** Override */
  virtual const Scene_Ptr& get_scene();

  /**
   * \brief Gets the current reconstructed scene.
   *
   * \return  The current reconstructed scene.
   */
  Scene_CPtr get_scene() const;

  /**
   * \brief Gets the settings to use for InfiniTAM.
   *
   * \return  The settings to use for InfiniTAM.
   */
  const Settings_CPtr& get_settings() const;

  /**
   * \brief Gets the current tracking state.
   *
   * \return  The current tracking state.
   */
  const TrackingState_Ptr& get_tracking_state();

  /**
   * \brief Gets the current tracking state.
   *
   * \return  The current tracking state.
   */
  TrackingState_CPtr get_tracking_state() const;

  /** Override */
  virtual const View_Ptr& get_view();

  /**
   * \brief Gets the current view of the scene.
   *
   * \return  The current view of the scene.
   */
  View_CPtr get_view() const;

  /** Override */
  virtual VisualisationEngine_CPtr get_visualisation_engine() const;

  /** Override */
  virtual void set_view(ITMLib::ITMView *view);
};

//#################### TYPEDEFS ####################

typedef boost::shared_ptr<Model> Model_Ptr;
typedef boost::shared_ptr<const Model> Model_CPtr;

#endif
