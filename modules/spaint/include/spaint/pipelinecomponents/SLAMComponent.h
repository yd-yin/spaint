/**
 * spaint: SLAMComponent.h
 * Copyright (c) Torr Vision Group, University of Oxford, 2016. All rights reserved.
 */

#ifndef H_SPAINT_SLAMCOMPONENT
#define H_SPAINT_SLAMCOMPONENT

#include <ITMLib/Core/ITMDenseMapper.h>
#include <ITMLib/Core/ITMDenseSurfelMapper.h>

#include "SLAMContext.h"
#include "../fiducials/FiducialDetector.h"
#include "../trackers/FallibleTracker.h"

namespace spaint {

/**
 * \brief An instance of this pipeline component can be used to perform simultaneous localisation and mapping (SLAM).
 */
class SLAMComponent
{
  //#################### TYPEDEFS ####################
private:
  typedef boost::shared_ptr<ITMLib::ITMDenseMapper<SpaintVoxel,ITMVoxelIndex> > DenseMapper_Ptr;
  typedef boost::shared_ptr<ITMLib::ITMDenseSurfelMapper<SpaintSurfel> > DenseSurfelMapper_Ptr;
  typedef ITMLib::ITMTrackingState::TrackingResult TrackingResult;

  //#################### ENUMERATIONS ####################
public:
  /**
   * \brief The values of this enumeration denote the different mapping modes that can be used by a SLAM component.
   */
  enum MappingMode
  {
    /** Produce both voxel and surfel maps. */
    MAP_BOTH,

    /** Produce only a voxel map. */
    MAP_VOXELS_ONLY
  };

  /**
   * \brief The values of this enumeration denote the different tracking modes that can be used by a SLAM component.
   */
  enum TrackingMode
  {
    /** Track against the surfel map. */
    TRACK_SURFELS,

    /** Track against the voxel map. */
    TRACK_VOXELS
  };

  //#################### PRIVATE VARIABLES ####################
private:
  /** The shared context needed for SLAM. */
  SLAMContext_Ptr m_context;

  /** The dense surfel mapper. */
  DenseSurfelMapper_Ptr m_denseSurfelMapper;

  /** The dense voxel mapper. */
  DenseMapper_Ptr m_denseVoxelMapper;

  /** Whether or not the user wants fiducials to be detected. */
  bool m_detectFiducials;

  /** A pointer to a tracker that can detect tracking failures (if available). */
  FallibleTracker *m_fallibleTracker;

  /** The fiducial detector to use (if any). */
  FiducialDetector_CPtr m_fiducialDetector;

  /** The number of frames for which fusion has been run. */
  size_t m_fusedFramesCount;

  /** Whether or not the user wants fusion to be run. */
  bool m_fusionEnabled;

  /** The engine used to provide input images to the fusion process. */
  ImageSourceEngine_Ptr m_imageSourceEngine;

  /** The IMU calibrator. */
  IMUCalibrator_Ptr m_imuCalibrator;

  /**
   * A number of initial frames to fuse, regardless of their tracking quality.
   * Tracking quality can be poor in the first few frames, when there is only
   * a limited model against which to track. By forcibly fusing these frames,
   * we prevent poor tracking quality from stopping the reconstruction. After
   * these frames have been fused, only frames with a good tracking result will
   * be fused.
   */
  size_t m_initialFramesToFuse;

  /** The engine used to perform low-level image processing operations. */
  LowLevelEngine_Ptr m_lowLevelEngine;

  /** The mapping mode to use. */
  MappingMode m_mappingMode;

  /** The ID of the scene (if any) whose pose is to be mirrored. */
  std::string m_mirrorSceneID;

  /** Whether or not to relocalise and train after processing every frame, for evaluation purposes. */
  bool m_relocaliseEveryFrame;

  /** The path to the relocalisation forest. */
  std::string m_relocaliserForestPath;

  /** The type of relocaliser. */
  std::string m_relocaliserType;

  /** The ID of the scene to reconstruct. */
  std::string m_sceneID;

  /** The tracker. */
  Tracker_Ptr m_tracker;

  /** The tracker configuration to use (in XML format). */
  std::string m_trackerConfig;

  /** The tracking controller. */
  TrackingController_Ptr m_trackingController;

  /** The tracking mode to use. */
  TrackingMode m_trackingMode;

  /** The view builder. */
  ViewBuilder_Ptr m_viewBuilder;

  //#################### CONSTRUCTORS ####################
public:
  /**
   * \brief Constructs a SLAM component.
   *
   * \param context           The shared context needed for SLAM.
   * \param sceneID           The ID of the scene to reconstruct.
   * \param imageSourceEngine The engine used to provide input images to the fusion process.
   * \param trackerConfig     The tracker configuration to use.
   * \param mappingMode       The mapping mode to use.
   * \param trackingMode      The tracking mode to use.
   * \param fiducialDetector  The fiducial detector to use (if any).
   * \param detectFiducials   Whether or not to initially detect fiducials in the scene.
   */
  SLAMComponent(const SLAMContext_Ptr& context, const std::string& sceneID, const ImageSourceEngine_Ptr& imageSourceEngine,
                const std::string& trackerConfig, MappingMode mappingMode = MAP_VOXELS_ONLY, TrackingMode trackingMode = TRACK_VOXELS,
                const FiducialDetector_CPtr& fiducialDetector = FiducialDetector_CPtr(), bool detectFiducials = false);

  //#################### PUBLIC MEMBER FUNCTIONS ####################
public:
  /**
   * \brief Gets whether or not the user wants fusion to be run.
   *
   * \return  true, if the user wants fusion to be run, or false otherwise.
   */
  bool get_fusion_enabled() const;

  /**
   * \brief Makes the SLAM component mirror the pose of the specified scene, rather than using its own tracker.
   *
   * \param mirrorSceneID The ID of the scene whose pose is to be mirrored.
   */
  void mirror_pose_of(const std::string& mirrorSceneID);

  /**
   * \brief Attempts to run the SLAM component for a single frame.
   *
   * \return  true, if a frame was processed, or false otherwise.
   */
  bool process_frame();

  /**
   * \brief Resets the reconstructed scene.
   */
  void reset_scene();

  /**
   * \brief Sets whether or not the user wants fiducials to be detected.
   *
   * \param detectFiducials Whether or not the user wants fiducials to be detected.
   */
  void set_detect_fiducials(bool detectFiducials);

  /**
   * \brief Sets whether or not the user wants fusion to be run.
   *
   * Note: Just because the user wants fusion to be run doesn't mean that it necessarily will be on every frame.
   *       In particular, we prevent fusion when we know we have lost tracking, regardless of this setting.
   *
   * \param fusionEnabled Whether or not the user wants fusion to be run.
   */
  void set_fusion_enabled(bool fusionEnabled);

  //#################### PRIVATE MEMBER FUNCTIONS ####################
private:
  /**
   * \brief Render from the live camera position to prepare for tracking.
   *
   * \param trackingMode  The tracking mode to use.
   */
  void prepare_for_tracking(TrackingMode trackingMode);

  /**
   * \brief Perform relocalisation-specific operations (i.e. train a relocaliser if tracking succeeded or relocalise otherwise).
   */
  void process_relocalisation();

  /**
   * \brief Sets up the relocaliser.
   */
  void setup_relocaliser();

  /**
   * \brief Sets up the tracker.
   */
  void setup_tracker();
};

//#################### TYPEDEFS ####################

typedef boost::shared_ptr<SLAMComponent> SLAMComponent_Ptr;

}

#endif
