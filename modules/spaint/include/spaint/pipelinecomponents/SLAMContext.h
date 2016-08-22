/**
 * spaint: SLAMContext.h
 * Copyright (c) Torr Vision Group, University of Oxford, 2016. All rights reserved.
 */

#ifndef H_SPAINT_SLAMCONTEXT
#define H_SPAINT_SLAMCONTEXT

#include <boost/shared_ptr.hpp>

#include <ITMLib/Engines/Visualisation/Interface/ITMVisualisationEngine.h>

#include "../util/ITMImagePtrTypes.h"
#include "../util/SpaintVoxel.h"

namespace spaint {

/**
* \brief An instance of a class deriving from this one provides the shared context needed by a SLAM component.
*/
class SLAMContext
{
  //#################### TYPEDEFS ####################
private:
  typedef ITMLib::ITMScene<spaint::SpaintVoxel,ITMVoxelIndex> Scene;
  typedef boost::shared_ptr<Scene> Scene_Ptr;
  typedef boost::shared_ptr<const Scene> Scene_CPtr;
  typedef boost::shared_ptr<ITMLib::ITMTrackingState> TrackingState_Ptr;
  typedef boost::shared_ptr<const ITMLib::ITMTrackingState> TrackingState_CPtr;
  typedef boost::shared_ptr<ITMLib::ITMView> View_Ptr;
  typedef boost::shared_ptr<const ITMLib::ITMView> View_CPtr;
  typedef boost::shared_ptr<const ITMLib::ITMVisualisationEngine<SpaintVoxel,ITMVoxelIndex> > VisualisationEngine_CPtr;

  //#################### PRIVATE VARIABLES ####################
private:
  /** The image into which depth input is read each frame. */
  ITMShortImage_Ptr m_inputRawDepthImage;

  /** The image into which RGB input is read each frame. */
  ITMUChar4Image_Ptr m_inputRGBImage;

  /** The current reconstructed scene. */
  Scene_Ptr m_scene;

  /** The current tracking state (containing the camera pose and additional tracking information used by InfiniTAM). */
  TrackingState_Ptr m_trackingState;

  /** The current view of the scene. */
  View_Ptr m_view;

  //#################### DESTRUCTOR ####################
public:
  /**
   * \brief Destroys the SLAM context.
   */
  virtual ~SLAMContext() {}

  //#################### PUBLIC ABSTRACT MEMBER FUNCTIONS ####################
public:
  virtual VisualisationEngine_CPtr get_visualisation_engine() const = 0;

  //#################### PUBLIC MEMBER FUNCTIONS ####################
public:
  /**
   * \brief Gets the dimensions of the depth images from which the scene is being reconstructed.
   *
   * \return  The dimensions of the depth images from which the scene is being reconstructed.
   */
  virtual const Vector2i& get_depth_image_size() const;

  /**
   * \brief TODO
   */
  virtual const ITMShortImage_Ptr& get_input_raw_depth_image();

  /**
   * \brief TODO
   */
  virtual const ITMUChar4Image_Ptr& get_input_rgb_image();

  /**
   * \brief Gets the intrinsic parameters for the camera that is being used to reconstruct the scene.
   *
   * \return  The intrinsic parameters for the camera.
   */
  virtual const ITMLib::ITMIntrinsics& get_intrinsics() const;

  /**
   * \brief Gets the current pose of the camera that is being used to reconstruct the scene.
   *
   * \return  The current camera pose.
   */
  virtual const ORUtils::SE3Pose& get_pose() const;

  /**
   * \brief Gets the dimensions of the RGB images from which the scene is being reconstructed.
   *
   * \return  The dimensions of the RGB images from which the scene is being reconstructed.
   */
  virtual const Vector2i& get_rgb_image_size() const;

  /**
   * \brief Gets the reconstructed scene.
   *
   * \return  The reconstructed scene.
   */
  virtual const Scene_Ptr& get_scene();

  /**
   * \brief Gets the current reconstructed scene.
   *
   * \return  The current reconstructed scene.
   */
  virtual Scene_CPtr get_scene() const;

  /**
   * \brief Gets the current tracking state.
   *
   * \return  The current tracking state.
   */
  virtual const TrackingState_Ptr& get_tracking_state();

  /**
   * \brief Gets the current tracking state.
   *
   * \return  The current tracking state.
   */
  virtual TrackingState_CPtr get_tracking_state() const;

  /**
   * \brief Gets the current view of the scene.
   *
   * \return  The current view of the scene.
   */
  virtual const View_Ptr& get_view();

  /**
   * \brief Gets the current view of the scene.
   *
   * \return  The current view of the scene.
   */
  virtual View_CPtr get_view() const;

  //#################### PRIVATE MEMBER FUNCTIONS ####################
private:
  /**
   * \brief TODO
   */
  void set_input_raw_depth_image(ITMShortImage *inputRawDepthImage);

  /**
   * \brief TODO
   */
  virtual void set_input_rgb_image(ITMUChar4Image *inputRGBImage);

  /**
   * \brief TODO
   */
  void set_scene(Scene *scene);

  /**
   * \brief TODO
   */
  void set_tracking_state(ITMLib::ITMTrackingState *trackingState);

  /**
   * \brief Sets the current view of the scene.
   *
   * \param view  The new current view of the scene.
   */
  void set_view(ITMLib::ITMView *view);

  //#################### FRIENDS ####################

  friend class SLAMComponent;
};

//#################### TYPEDEFS ####################

typedef boost::shared_ptr<SLAMContext> SLAMContext_Ptr;

}

#endif
