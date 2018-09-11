/**
 * itmx: ICPRefiningRelocaliser.tpp
 * Copyright (c) Torr Vision Group, University of Oxford, 2017. All rights reserved.
 */

#include "ICPRefiningRelocaliser.h"

#include <iostream>
#include <stdexcept>

#include <ITMLib/Core/ITMTrackingController.h>
#include <ITMLib/Engines/Visualisation/ITMVisualisationEngineFactory.h>
#include <ITMLib/Objects/RenderStates/ITMRenderStateFactory.h>
#include <ITMLib/Trackers/ITMTrackerFactory.h>

#include <orx/base/ORImagePtrTypes.h>

#include <tvgutil/filesystem/PathFinder.h>
#include <tvgutil/misc/SettingsContainer.h>
#include <tvgutil/timing/TimeUtil.h>

#include "../persistence/PosePersister.h"
#include "../visualisation/DepthVisualisationUtil.tpp"
#include "../visualisation/DepthVisualiserFactory.h"

#ifdef WITH_OPENCV
#include "../ocv/OpenCVUtil.h"
#endif

namespace itmx {

//#################### CONSTRUCTORS ####################

template <typename VoxelType, typename IndexType>
ICPRefiningRelocaliser<VoxelType,IndexType>::ICPRefiningRelocaliser(const orx::Relocaliser_Ptr& innerRelocaliser, const Tracker_Ptr& tracker,
                                                                    const Vector2i& rgbImageSize, const Vector2i& depthImageSize,
                                                                    const ITMLib::ITMRGBDCalib& calib, const Scene_Ptr& scene,
                                                                    const DenseMapper_Ptr& denseVoxelMapper, const Settings_CPtr& settings)
: RefiningRelocaliser(innerRelocaliser),
  m_denseVoxelMapper(denseVoxelMapper),
  m_depthVisualiser(DepthVisualiserFactory::make_depth_visualiser(settings->deviceType)),
  m_scene(scene),
  m_settings(settings),
  m_timerInitialRelocalisation("Initial Relocalisation"),
  m_timerRefinement("ICP Refinement"),
  m_timerRelocalisation("Relocalisation"),
  m_timerTraining("Training"),
  m_timerUpdate("Update"),
  m_tracker(tracker),
  m_visualisationEngine(ITMVisualisationEngineFactory::MakeVisualisationEngine<VoxelType,IndexType>(settings->deviceType))
{
  // Construct the tracking controller, tracking state, view and render state.
  m_trackingController.reset(new ITMLib::ITMTrackingController(m_tracker.get(), m_settings.get()));
  m_trackingState.reset(new ITMLib::ITMTrackingState(depthImageSize, m_settings->GetMemoryType()));
  m_view.reset(new ITMLib::ITMView(calib, rgbImageSize, depthImageSize, m_settings->deviceType == DEVICE_CUDA));
  m_voxelRenderState.reset(ITMLib::ITMRenderStateFactory<IndexType>::CreateRenderState(
    m_trackingController->GetTrackedImageSize(rgbImageSize, depthImageSize),
    m_scene->sceneParams,
    m_settings->GetMemoryType()
  ));

  // Configure the relocaliser based on the settings that have been passed in.
  const static std::string settingsNamespace = "ICPRefiningRelocaliser.";
  m_chooseBestResult = m_settings->get_first_value<bool>(settingsNamespace + "chooseBestResult", false);
  m_savePoses = m_settings->get_first_value<bool>(settingsNamespace + "saveRelocalisationPoses", false);
  m_saveTimes = m_settings->get_first_value<bool>(settingsNamespace + "saveRelocalisationTimes", false);
  m_scorePercentile = m_settings->get_first_value<float>(settingsNamespace + "scorePercentile", 0.95f);
  m_timersEnabled = m_settings->get_first_value<bool>(settingsNamespace + "timersEnabled", false);

  // Get the (global) experiment tag.
  const std::string experimentTag = m_settings->get_first_value<std::string>("experimentTag", tvgutil::TimeUtil::get_iso_timestamp());

  if(m_savePoses)
  {
    // Determine the directory to which to save the poses and make sure that it exists.
    m_posePathGenerator.reset(tvgutil::SequentialPathGenerator(tvgutil::find_subdir_from_executable("reloc_poses") / experimentTag));
    boost::filesystem::create_directories(m_posePathGenerator->get_base_dir());

    // Output the directory we're using (for debugging purposes).
    std::cout << "Saving relocalisation poses in: " << m_posePathGenerator->get_base_dir() << '\n';
  }

  if(m_saveTimes)
  {
    // Forcefully enable timers.
    m_timersEnabled = true;

    // Make sure the directory where we want to save the relocalisation times exists.
    boost::filesystem::path timersOutputFolder(tvgutil::find_subdir_from_executable("reloc_times"));
    boost::filesystem::create_directories(timersOutputFolder);

    // Prepare the output filename.
    m_timersOutputFile = (timersOutputFolder / (experimentTag + ".txt")).string();
  }
}

//#################### DESTRUCTOR ####################

template <typename VoxelType, typename IndexType>
ICPRefiningRelocaliser<VoxelType,IndexType>::~ICPRefiningRelocaliser()
{
  if(m_timersEnabled)
  {
    std::cout << "Training calls: " << m_timerTraining.count() << ", average duration: " << m_timerTraining.average_duration() << '\n';
    std::cout << "Update calls: " << m_timerUpdate.count() << ", average duration: " << m_timerUpdate.average_duration() << '\n';
    std::cout << "Initial Relocalisation calls: " << m_timerInitialRelocalisation.count() << ", average duration: " << m_timerInitialRelocalisation.average_duration() << '\n';
    std::cout << "ICP Refinement calls: " << m_timerRefinement.count() << ", average duration: " << m_timerRefinement.average_duration() << '\n';
    std::cout << "Total Relocalisation calls: " << m_timerRelocalisation.count() << ", average duration: " << m_timerRelocalisation.average_duration() << '\n';
  }

  if(m_saveTimes)
  {
    std::cout << "Saving relocalisation average times in: " << m_timersOutputFile << "\n";
    std::ofstream out(m_timersOutputFile.c_str());

    // Output the average durations.
    out << m_timerTraining.average_duration().count() << " "
        << m_timerUpdate.average_duration().count() << " "
        << m_timerInitialRelocalisation.average_duration().count() << " "
        << m_timerRefinement.average_duration().count() << " "
        << m_timerRelocalisation.average_duration().count() << "\n";
  }
}

//#################### PUBLIC MEMBER FUNCTIONS ####################

template <typename VoxelType, typename IndexType>
void ICPRefiningRelocaliser<VoxelType,IndexType>::finish_training()
{
  m_innerRelocaliser->finish_training();
}

template <typename VoxelType, typename IndexType>
void ICPRefiningRelocaliser<VoxelType,IndexType>::load_from_disk(const std::string& inputFolder)
{
  m_innerRelocaliser->load_from_disk(inputFolder);
}

template <typename VoxelType, typename IndexType>
std::vector<orx::Relocaliser::Result>
ICPRefiningRelocaliser<VoxelType, IndexType>::relocalise(const ORUChar4Image *colourImage, const ORFloatImage *depthImage,
                                                         const Vector4f& depthIntrinsics) const
{
  std::vector<ORUtils::SE3Pose> initialPoses;
  return relocalise(colourImage, depthImage, depthIntrinsics, initialPoses);
}

template <typename VoxelType, typename IndexType>
std::vector<orx::Relocaliser::Result>
ICPRefiningRelocaliser<VoxelType, IndexType>::relocalise(const ORUChar4Image *colourImage, const ORFloatImage *depthImage,
                                                         const Vector4f& depthIntrinsics, std::vector<ORUtils::SE3Pose>& initialPoses) const
{
#if DEBUGGING
  static int frameIdx = -1;
  ++frameIdx;
  std::cout << "---\nFrame Index: " << frameIdx << std::endl;
#endif

  // Reset the initial poses.
  initialPoses.clear();

  // Run the inner relocaliser. If it fails, save dummy poses and early out.
  start_timer(m_timerRelocalisation);
  start_timer(m_timerInitialRelocalisation, false); // No need to synchronize the GPU again.
  std::vector<Result> initialResults = m_innerRelocaliser->relocalise(colourImage, depthImage, depthIntrinsics);
  stop_timer(m_timerInitialRelocalisation);

  if(initialResults.empty())
  {
    Matrix4f invalidPose;
    invalidPose.setValues(std::numeric_limits<float>::quiet_NaN());
    save_poses(invalidPose, invalidPose);
    stop_timer(m_timerRelocalisation, false); // No need to synchronize the GPU again.
    return std::vector<Relocaliser::Result>();
  }

  std::vector<Relocaliser::Result> refinedResults;
  float bestScore = static_cast<float>(INT_MAX);

  start_timer(m_timerRefinement, false); // No need to synchronize the GPU again.

  // Reset the render state before raycasting (we do it once for relocalisation attempt).
  // FIXME: It would be nicer to simply reuse it, but unfortunately this leads
  //        to the program randomly crashing after a while. The crash may be occurring because we don't use this render
  //        state to integrate frames into the scene, but we haven't been able to pin this down yet. As a result, we
  //        currently reset it each time as a workaround. A mildly less costly alternative might
  //        be to pass in a render state that is being used elsewhere and reuse it here, but that feels messier.
  m_voxelRenderState->Reset();

  // For each initial result from the inner relocaliser:
  for(size_t resultIdx = 0; resultIdx < initialResults.size(); ++resultIdx)
  {
    // Get the suggested pose.
    const ORUtils::SE3Pose initialPose = initialResults[resultIdx].pose;

    // Copy the depth and RGB images into the view.
    m_view->depth->SetFrom(depthImage, m_settings->deviceType == DEVICE_CUDA ? ORFloatImage::CUDA_TO_CUDA : ORFloatImage::CPU_TO_CPU);
    m_view->rgb->SetFrom(colourImage, m_settings->deviceType == DEVICE_CUDA ? ORUChar4Image::CUDA_TO_CUDA : ORUChar4Image::CPU_TO_CPU);

    // Set up the tracking state using the initial pose.
    m_trackingState->pose_d->SetFrom(&initialPose);

    // Update the list of visible blocks.
    const bool resetVisibleList = true;
    m_denseVoxelMapper->UpdateVisibleList(m_view.get(), m_trackingState.get(), m_scene.get(), m_voxelRenderState.get(), resetVisibleList);

    // Raycast from the initial pose to prepare for tracking.
    m_trackingController->Prepare(m_trackingState.get(), m_scene.get(), m_view.get(), m_visualisationEngine.get(), m_voxelRenderState.get());

    // Run the tracker to refine the initial pose.
    m_trackingController->Track(m_trackingState.get(), m_view.get());

    // If tracking succeeded:
    if(m_trackingState->trackerResult != ITMLib::ITMTrackingState::TRACKING_FAILED)
    {
      // Set up the refined result.
      Result refinedResult;
      refinedResult.pose.SetFrom(m_trackingState->pose_d);
      refinedResult.quality = m_trackingState->trackerResult == ITMLib::ITMTrackingState::TRACKING_GOOD ? RELOCALISATION_GOOD : RELOCALISATION_POOR;
      refinedResult.score = m_trackingState->trackerScore;

      // If we're trying to choose the best relocalisation after refinement:
      if(m_chooseBestResult)
      {
        // Score the refined result.
        refinedResult.score = score_result(refinedResult);

#if DEBUGGING
        std::cout << resultIdx << ": " << refinedResult.score << '\n';
#endif

        // If the score is better than the current best score, update the current best score and result.
        if(refinedResult.score < bestScore)
        {
          bestScore = refinedResult.score;
          initialPoses.clear();
          initialPoses.push_back(initialPose);
          refinedResults.clear();
          refinedResults.push_back(refinedResult);
        }
      }
      else
      {
        // If we're not trying to choose the best relocalisation,
        // simply store the initial pose and refined result without any scoring.
        initialPoses.push_back(initialPose);
        refinedResults.push_back(refinedResult);
      }
    }
  }

  stop_timer(m_timerRefinement);
  stop_timer(m_timerRelocalisation, false); // No need to synchronize the GPU again.

  // Save the best initial and refined poses if needed.
  if(m_savePoses)
  {
    // The best initial pose is the best one returned by the inner relocaliser.
    const Matrix4f initialPose = initialResults[0].pose.GetInvM();

    // The best refined pose is the pose (if any) whose score is lowest after refinement. Note that the
    // best refined pose is not necessarily the result of refining the best initial pose, since refinement
    // is not monotonic. If refinement failed for all initial poses, the refined pose is set to NaNs.
    Matrix4f refinedPose;
    if(!refinedResults.empty())
    {
      refinedPose = refinedResults[0].pose.GetInvM();
    }
    else
    {
      refinedPose.setValues(std::numeric_limits<float>::quiet_NaN());
    }

    // Actually save the poses.
    save_poses(initialPose, refinedPose);

    // Since we are saving the poses (i.e. we are running in evaluation mode), we force the quality of
    // every refined result to POOR to prevent fusion whilst evaluating the testing sequence.
    for(size_t i = 0; i < refinedResults.size(); ++i)
    {
      refinedResults[i].quality = RELOCALISATION_POOR;
    }
  }

  return refinedResults;
}

template <typename VoxelType, typename IndexType>
void ICPRefiningRelocaliser<VoxelType,IndexType>::reset()
{
  m_innerRelocaliser->reset();
}

template <typename VoxelType, typename IndexType>
void ICPRefiningRelocaliser<VoxelType,IndexType>::save_to_disk(const std::string& outputFolder) const
{
  m_innerRelocaliser->save_to_disk(outputFolder);
}

template <typename VoxelType, typename IndexType>
void ICPRefiningRelocaliser<VoxelType,IndexType>::train(const ORUChar4Image *colourImage, const ORFloatImage *depthImage,
                                                        const Vector4f& depthIntrinsics, const ORUtils::SE3Pose& cameraPose)
{
  start_timer(m_timerTraining);
  m_innerRelocaliser->train(colourImage, depthImage, depthIntrinsics, cameraPose);
  stop_timer(m_timerTraining);
}

template <typename VoxelType, typename IndexType>
void ICPRefiningRelocaliser<VoxelType,IndexType>::update()
{
  start_timer(m_timerUpdate);
  m_innerRelocaliser->update();
  stop_timer(m_timerUpdate);
}

//#################### PRIVATE MEMBER FUNCTIONS ####################

template <typename VoxelType, typename IndexType>
void ICPRefiningRelocaliser<VoxelType,IndexType>::save_poses(const Matrix4f& relocalisedPose, const Matrix4f& refinedPose) const
{
  if(!m_savePoses) return;

  PosePersister::save_pose_on_thread(relocalisedPose, m_posePathGenerator->make_path("pose-%06i.reloc.txt"));
  PosePersister::save_pose_on_thread(refinedPose, m_posePathGenerator->make_path("pose-%06i.icp.txt"));
  m_posePathGenerator->increment_index();
}

template <typename VoxelType, typename IndexType>
float ICPRefiningRelocaliser<VoxelType,IndexType>::score_result(const Result& result) const
{
#ifdef WITH_OPENCV
  // Make an OpenCV wrapper of the current depth image.
  m_view->depth->UpdateHostFromDevice();
  cv::Mat cvRealDepth(m_view->depth->noDims.y, m_view->depth->noDims.x, CV_32FC1, m_view->depth->GetData(MEMORYDEVICE_CPU));

  // Render a synthetic depth image of the scene from the suggested pose.
  ORFloatImage_Ptr synthDepth(new ORFloatImage(m_view->depth->noDims, true, true));
  DepthVisualisationUtil<VoxelType,IndexType>::generate_depth_from_voxels(
    synthDepth, m_scene, result.pose, m_view->calib.intrinsics_d, m_voxelRenderState,
    DepthVisualiser::DT_ORTHOGRAPHIC, m_visualisationEngine, m_depthVisualiser, m_settings
  );

  // Make an OpenCV wrapper of the synthetic depth image.
  cv::Mat cvSynthDepth(synthDepth->noDims.y, synthDepth->noDims.x, CV_32FC1, synthDepth->GetData(MEMORYDEVICE_CPU));

#if DEBUGGING
  // If we're debugging, show the real and synthetic depth images to the user (note that we need to convert them to unsigned chars for visualisation).
  // We don't use the OpenCV normalize function because we want consistent visualisations for different frames (even though there might be clamping).
  const float scaleFactor = 255.0f / 5.0f;
  cv::Mat cvRealDepthVis, cvSynthDepthVis;
  cvRealDepth.convertTo(cvRealDepthVis, CV_8U, scaleFactor);
  cvSynthDepth.convertTo(cvSynthDepthVis, CV_8U, scaleFactor);

  cv::imshow("Real Depth", cvRealDepthVis);
  cv::imshow("Synthetic Depth", cvSynthDepthVis);
#endif

  // Compute a binary mask showing which pixels are valid in both the real and synthetic depth images.
  cv::Mat cvRealMask = cvRealDepth > 0;
  cv::Mat cvSynthMask = cvSynthDepth > 0;
  cv::Mat cvCombinedMask = cvRealMask & cvSynthMask;
  cv::erode(cvCombinedMask, cvCombinedMask, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7,7)));

#define USE_DIFFERENCE

#ifdef USE_DIFFERENCE
  // Compute the difference between the real and synthetic depth images, and mask it using the combined mask.
  cv::Mat cvDepthDiff, cvMaskedDepthDiff;
  cv::divide(cvRealDepth, cvSynthDepth, cvDepthDiff);
  cv::absdiff(cvDepthDiff, cv::Scalar(1), cvDepthDiff);
//  cv::absdiff(cvRealDepth, cvSynthDepth, cvDepthDiff);
  cvDepthDiff.copyTo(cvMaskedDepthDiff, cvCombinedMask);

  std::vector<cv::Mat> images = { cvMaskedDepthDiff };
  std::vector<int> channels = { 0 };
  std::vector<int> histSize = { 101 };
  std::vector<float> ranges = { 0.0f, 1.0f };
  cv::Mat histogram;

  // Histogram the distribution of errors.
  cv::calcHist(images, channels, cvCombinedMask, histogram, histSize, ranges);

  // Count how many valid pixels are in the combined mask.
  int validDepthsCount = cv::countNonZero(cvCombinedMask);

  // We are interested in a certain percentile.
  int percentileCount = m_scorePercentile * validDepthsCount;

  // Find the right bin.
  float accumulator = 0;
  int bin = 0;
  for(; bin < histogram.size().area(); ++bin)
  {
    accumulator += histogram.ptr<float>()[bin];
    if(accumulator > percentileCount) break;
  }

  float percentileError = bin * 0.01f;

//  std::cout << "95% percentile of " << validDepthsCount << " is: " << percentileCount
//            << " - counted: " << accumulator << " at bin: " << bin << " equals error: " << percentileError << std::endl;

#else
  cv::Mat cvDepthTargetRatio;
  cv::divide(cvRealDepth, cvSynthDepth, cvDepthTargetRatio);

  cv::Mat cvDepthInRangeMask;
  cv::inRange(cvDepthTargetRatio, cv::Scalar(0.975), cv::Scalar(1.025), cvDepthInRangeMask);

  cv::Scalar depthInRangeCount = cv::countNonZero(cvDepthInRangeMask);
  cv::Scalar validPixelsCount = cv::countNonZero(cvCombinedMask);

  double depthInRangeFraction = std::min(1.0, depthInRangeCount(0) / validPixelsCount(0));

#if 1 || DEBUGGING
  std::cout << "\nDepth in range fraction: " << depthInRangeFraction << std::endl;
#endif
#endif

#if DEBUGGING
  // We need to convert the image for visualisation.
  cv::Mat cvMaskedDepthDiffVis = cvMaskedDepthDiff / 0.05 * 255.0;
  cvMaskedDepthDiffVis.convertTo(cvMaskedDepthDiffVis, CV_8U);

  cv::imshow("Masked Depth Difference", cvMaskedDepthDiffVis);
  cv::waitKey(1);
#endif

#ifdef USE_DIFFERENCE
  // Determine the mean depth difference for valid pixels in the real and synthetic depth images.
  cv::Scalar meanDepthDiff, stdDevDepthDiff;
  cv::meanStdDev(cvMaskedDepthDiff, meanDepthDiff, stdDevDepthDiff, cvCombinedMask);

#if DEBUGGING
  std::cout << "\nMean Depth Difference: " << meanDepthDiff(0) << " - StdDev: " << stdDevDepthDiff(0) << std::endl;
#endif

  // Compute the fraction of the synthetic depth image that is valid.
  float validFraction = static_cast<float>(cv::countNonZero(cvSynthMask)) / (cvSynthMask.size().area());

#if DEBUGGING
  std::cout << "Valid Synthetic Depth Pixels: " << cv::countNonZero(cvSynthMask) << std::endl;
#endif

//  return validFraction >= 0.1f ? static_cast<float>(meanDepthDiff(0)) : static_cast<float>(INT_MAX);
  return validFraction >= 0.1f ? percentileError : static_cast<float>(INT_MAX);
#else
  // Compute the fraction of the synthetic depth image that is valid.
  float validFraction = static_cast<float>(cv::countNonZero(cvSynthMask)) / (cvSynthMask.size().area());

  return validFraction >= 0.1f ? static_cast<float>(depthInRangeFraction) : 0.0f;
#endif
#else
  return 0.0f;
#endif
}

template <typename VoxelType, typename IndexType>
void ICPRefiningRelocaliser<VoxelType,IndexType>::start_timer(AverageTimer& timer, bool cudaSynchronize) const
{
  if(!m_timersEnabled) return;

#ifdef WITH_CUDA
  if(cudaSynchronize)
  {
    ORcudaSafeCall(cudaDeviceSynchronize());
  }
#endif

  timer.start();
}

template <typename VoxelType, typename IndexType>
void ICPRefiningRelocaliser<VoxelType,IndexType>::stop_timer(AverageTimer& timer, bool cudaSynchronize) const
{
  if(!m_timersEnabled) return;

#ifdef WITH_CUDA
  if(cudaSynchronize)
  {
    ORcudaSafeCall(cudaDeviceSynchronize());
  }
#endif

  timer.stop();
}

}
