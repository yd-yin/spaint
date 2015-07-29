/**
 * spaint: RobustViconTracker.cpp
 * Copyright (c) Torr Vision Group, University of Oxford, 2015. All rights reserved.
 */

#include "trackers/RobustViconTracker.h"

#include <ITMLib/Engine/ITMTrackerFactory.h>
using namespace ITMLib::Engine;

#include "util/CameraPoseConverter.h"
using namespace rigging;

namespace spaint {

//#################### CONSTRUCTORS ####################

RobustViconTracker::RobustViconTracker(const std::string& host, const std::string& subjectName, const Vector2i& trackedImageSize, const Settings_CPtr& settings,
                                       const LowLevelEngine_CPtr& lowLevelEngine, const Scene_Ptr& scene)
{
  m_viconTracker.reset(new ViconTracker(host, subjectName));
  m_icpTracker.reset(ITMTrackerFactory<SpaintVoxel,ITMVoxelIndex>::Instance().Make(
    trackedImageSize, settings.get(), lowLevelEngine.get(), NULL, scene.get()
  ));
}

//#################### PUBLIC MEMBER FUNCTIONS ####################

void RobustViconTracker::TrackCamera(ITMTrackingState *trackingState, const ITMView *view)
{
  // Record the initial pose.
  Matrix4f initialM = trackingState->pose_d->GetM();
  SimpleCamera initialCam = CameraPoseConverter::pose_to_camera(*trackingState->pose_d);

  // Obtain a coarse pose using the Vicon tracker.
  m_viconTracker->TrackCamera(trackingState, view);

  // Record the Vicon pose.
  Matrix4f viconM = trackingState->pose_d->GetM();
  SimpleCamera viconCam = CameraPoseConverter::pose_to_camera(*trackingState->pose_d);

  // Attempt to refine the Vicon pose using ICP.
  m_icpTracker->TrackCamera(trackingState, view);

  // Check whether or not the ICP pose is acceptable by comparing it to the initial pose and Vicon pose.
  // Testing against the initial pose allows us to prevent fusion when the camera is moving too fast,
  // whilst testing against the Vicon pose allows us to prevent fusion when the ICP fails.
  SimpleCamera icpCam = CameraPoseConverter::pose_to_camera(*trackingState->pose_d);
  m_lostTracking = !poses_are_similar(initialCam, icpCam) || !poses_are_similar(viconCam, icpCam);
}

//#################### PRIVATE STATIC MEMBER FUNCTIONS ####################

double RobustViconTracker::angle_between(const Eigen::Vector3f& v1, const Eigen::Vector3f& v2)
{
  return acos(v1.dot(v2) / (v1.norm() * v2.norm()));
}

bool RobustViconTracker::poses_are_similar(const SimpleCamera& cam1, const SimpleCamera& cam2, double distanceThreshold, double angleThreshold)
{
  // Calculate the distance between the positions of the two cameras, and the angles between their corresponding camera axes.
  double pDist = (cam1.p() - cam2.p()).norm();
  double nAngle = angle_between(cam1.n(), cam2.n());
  double uAngle = angle_between(cam1.u(), cam2.u());
  double vAngle = angle_between(cam1.v(), cam2.v());

  // Check the similarity of the camera poses by comparing the aforementioned distance and angles to suitable thresholds.
  return pDist < distanceThreshold && nAngle < angleThreshold && uAngle < angleThreshold && vAngle < angleThreshold;
}

}
