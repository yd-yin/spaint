/**
 * relocgui: RelocaliserApplication.cpp
 * Copyright (c) Torr Vision Group, University of Oxford, 2017. All rights reserved.
 */

#include "RelocaliserApplication.h"
#include <unistd.h>
#include <iostream>
#include <stdexcept>

#include <boost/make_shared.hpp>

#include <Eigen/Geometry>

#include <grove/relocalisation/ScoreRelocaliserFactory.h>

#include <ITMLib/Engines/ViewBuilding/ITMViewBuilderFactory.h>
#include <ITMLib/Objects/Camera/ITMCalibIO.h>

#include <itmx/persistence/PosePersister.h>

#include <orx/base/MemoryBlockFactory.h>

#include <tvgutil/filesystem/PathFinder.h>
#include <tvgutil/timing/AverageTimer.h>
#include <tvgutil/timing/TimeUtil.h>

#include <fstream> 
#include <math.h> 

namespace bf = boost::filesystem;

using namespace ITMLib;
using namespace ORUtils;

using namespace grove;
using namespace itmx;
using namespace orx;
using namespace tvgutil;


bool g_b_valid_seq_rel = false;
std::set<int> g_valid_fids_rel = 
{
  4, 5, 6, 7, 9, 10, 
  11, 12, 13, 14, 16, 17, 19, 20, 21, 24, 25, 
  41, 43, 44, 45, 47, 48, 49, 50, 51, 52, 53, 54, 55, 
  56, 57, 58, 59, 60, 61, 63, 64, 70, 71, 72, 73, 74, 
  75, 76, 77, 78, 79, 80, 82, 89, 90, 91, 92, 94, 95, 
  97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 
  108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 
  119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 
  130, 131, 133, 134, 135, 136, 137, 138, 139, 140, 141, 
  142, 143, 144, 146, 148, 150, 151, 152, 153, 154, 155, 
  156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 
  167, 168, 169, 170, 171, 172, 177, 178, 180, 183, 185, 
  186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 
  197, 198, 199, 201, 202, 203, 204, 205, 206, 210, 213, 
  214, 215, 216, 218, 219, 220, 221, 222, 223, 224, 225, 
  227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 
  238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 
  249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260
};
std::vector<float> g_terr_rel;
std::vector<float> g_rerr_rel;
bool sort_func_rel(const float v1, const float v2)
{
  return v1<v2;
}
std::vector<int> g_ids_5cm5deg_rel;
std::vector<int> g_ids_10cm10deg_rel;
std::vector<int> g_ids_20cm10deg_rel;
int g_fid_rel = 1; 
int g_num_success_rel = 0;


namespace relocgui {

//#################### ANONYMOUS FREE FUNCTIONS ####################

namespace {

/**
 * \brief Computes the angular separation between two rotation matrices.
 *
 * \param r1 The first rotation matrix.
 * \param r2 The second rotation matrix.
 *
 * \return The angle of the transformation mapping r1 to r2.
 */
float angular_separation(const Eigen::Matrix3f &r1, const Eigen::Matrix3f &r2)
{
  // First calculate the rotation matrix which maps r1 to r2.
  Eigen::Matrix3f dr = r2 * r1.transpose();

  Eigen::AngleAxisf aa(dr);
  return aa.angle();
}

/**
 * \brief Check whether a pose matrix is similar enough to a ground truth pose matrix.
 *
 * \param gtPose               The ground truth pose matrix.
 * \param testPose             The candidate pose matrix.
 * \param translationMaxError  The maximum difference (in m) between the translation components of the two matrices.
 * \param angleMaxError        The maximum angular difference between the two rotation components (when converted to the
 *                             axis-angle representation).
 *
 * \return Whether or not the two poses differ for less or equal than translationMaxError and angleMaxError.
 */
bool pose_matches(const Matrix4f &gtPose, const Matrix4f &testPose, float translationMaxError, float angleMaxError)
{
  // Both our Matrix type and Eigen's are column major, so we can just use Map here.
  const Eigen::Map<const Eigen::Matrix4f> gtPoseEigen(gtPose.m);
  const Eigen::Map<const Eigen::Matrix4f> testPoseEigen(testPose.m);

  const Eigen::Matrix3f gtR = gtPoseEigen.block<3, 3>(0, 0);
  const Eigen::Matrix3f testR = testPoseEigen.block<3, 3>(0, 0);
  const Eigen::Vector3f gtT = gtPoseEigen.block<3, 1>(0, 3);
  const Eigen::Vector3f testT = testPoseEigen.block<3, 1>(0, 3);

  const float translationError = (gtT - testT).norm();
  const float angleError = angular_separation(gtR, testR);

  if ( 
    (g_b_valid_seq_rel && g_valid_fids_rel.find(g_fid_rel) != std::end(g_valid_fids_rel)) 
    || 
    (!g_b_valid_seq_rel) 
    )
  {
    float rel_terr = translationError*100;
    float rel_rerr = angleError / 3.14 * 180.0;
    if (std::isnan(rel_terr))
    {
      rel_terr = 999;
      rel_rerr = 999;
      printf("%.2f cm %.2f deg\n", rel_terr, rel_rerr);
      g_terr_rel.push_back(rel_terr);
      g_rerr_rel.push_back(rel_rerr);
    }
    else
    {
      printf("%.2f cm %.2f deg\n", rel_terr, rel_rerr);
      g_terr_rel.push_back(rel_terr);
      g_rerr_rel.push_back(rel_rerr);
    }
    printf("%d valid frames done.\n", g_terr_rel.size());
    std::sort(g_terr_rel.begin(), g_terr_rel.end(), sort_func_rel);
    std::sort(g_rerr_rel.begin(), g_rerr_rel.end(), sort_func_rel);
    if (g_terr_rel.size()%2==0)
    {
      int idx1 = g_terr_rel.size()/2-1;
      int idx2 = idx1 + 1;
      printf("median %.2f cm %.2f deg\n", (g_terr_rel[idx1]+g_terr_rel[idx2])/2, (g_rerr_rel[idx1]+g_rerr_rel[idx2])/2);
    }
    else
    {
      int idx = g_terr_rel.size()/2;
      printf("median %.2f cm %.2f deg\n", g_terr_rel[idx], g_rerr_rel[idx]);
    }
    if (translationError <= translationMaxError && angleError <= angleMaxError) g_num_success_rel++;
    printf("success = %.2f%%\n", (float)g_num_success_rel / g_terr_rel.size() * 100);
    printf("total %d frames, success %d frames\n", g_terr_rel.size(), g_num_success_rel);
    if (rel_terr<10 && rel_rerr<10) g_ids_10cm10deg_rel.push_back(g_terr_rel.size());
    printf("success(10cm10deg) = %.2f%%\n", (float)g_ids_10cm10deg_rel.size() / g_terr_rel.size() * 100);
  }
  g_fid_rel++;

  return translationError <= translationMaxError && angleError <= angleMaxError;
}

/**
 * \brief Read a 4x4 pose matrix from a text file.
 *
 * \param fileName Path to the text file.
 *
 * \return The pose matrix.
 *
 * \throws std::runtime_error if the file is missing or has the wrong format.
 */
Matrix4f read_pose_from_file(const bf::path &fileName)
{
  if(!bf::is_regular(fileName)) throw std::runtime_error("File not found: " + fileName.string());

  std::ifstream in(fileName.string().c_str());

  Matrix4f res;
  in >> res(0, 0) >> res(1, 0) >> res(2, 0) >> res(3, 0);
  in >> res(0, 1) >> res(1, 1) >> res(2, 1) >> res(3, 1);
  in >> res(0, 2) >> res(1, 2) >> res(2, 2) >> res(3, 2);
  in >> res(0, 3) >> res(1, 3) >> res(2, 3) >> res(3, 3);

  if(!in) throw std::runtime_error("Error reading a pose matrix from: " + fileName.string());

  return res;
}
}

//#################### CONSTRUCTOR ####################

RelocaliserApplication::RelocaliserApplication(const std::string &calibrationPath,
                                               const std::string &trainingPath,
                                               const std::string &testingPath,
                                               const SettingsContainer_CPtr &settings)
  : m_calibrationFilePath(calibrationPath)
  , m_saveRelocalisedPoses(true)
  , m_settingsContainer(settings)
  , m_testSequencePath(testingPath)
  , m_trainSequencePath(trainingPath)
{
  // Read camera calibration parameters.
  if(!ITMLib::readRGBDCalib(m_calibrationFilePath.string().c_str(), m_cameraCalibration))
  {
    throw std::invalid_argument("Couldn't read calibration parameters.");
  }

  // // Setup the image masks.
  // m_depthImageMask = m_settingsContainer->get_first_value<std::string>("depthImageMask", "frame-%06d.depth.png");
  // m_poseFileMask = m_settingsContainer->get_first_value<std::string>("poseFileMask", "frame-%06d.pose.txt");
  // m_rgbImageMask = m_settingsContainer->get_first_value<std::string>("rgbImageMask", "frame-%06d.color.png");

  // Create the folder that will store the relocalised poses.
  if(m_saveRelocalisedPoses)
  {
    const std::string experimentTag =
        m_settingsContainer->get_first_value<std::string>("experimentTag", TimeUtil::get_iso_timestamp());
    m_outputPosesPath = find_subdir_from_executable("reloc_poses") / experimentTag;

    // std::cout << "Saving output poses in: " << m_outputPosesPath << '\n';
    if(!bf::is_directory(m_outputPosesPath))
    {
      bf::create_directories(m_outputPosesPath);
    }
  }

  // Set up the path generators.
  m_outputPosesPathGenerator = boost::make_shared<SequentialPathGenerator>(m_outputPosesPath);
  m_testingSequencePathGenerator = boost::make_shared<SequentialPathGenerator>(m_testSequencePath);
  m_trainingSequencePathGenerator = boost::make_shared<SequentialPathGenerator>(m_trainSequencePath);

  // We try to run everything on the GPU.
  const DeviceType deviceType = DEVICE_CUDA;

  // Setup the relocaliser.
  const bf::path resourcesFolder = find_subdir_from_executable("resources");
  const bf::path defaultRelocalisationForestPath = resourcesFolder / "DefaultRelocalisationForest.rf"; // raw
  if(!g_phase.compare("test4rl") == 0)
  {
    std::cout << "Loading relocalisation forest from: " << defaultRelocalisationForestPath << '\n';
  }

  m_relocaliser = ScoreRelocaliserFactory::make_score_relocaliser(defaultRelocalisationForestPath.string(), m_settingsContainer, deviceType);

  // Create a ViewBuilder to convert the depth image to float (might to it with OpenCV as well but this is easier).
  m_viewBuilder.reset(ITMViewBuilderFactory::MakeViewBuilder(m_cameraCalibration, deviceType));

  // Allocate the ITM Images used to train/test the relocaliser (empty for now, will be resized later).
  const MemoryBlockFactory& mbf = MemoryBlockFactory::instance();
  m_currentDepthImage = mbf.make_image<float>();
  m_currentRawDepthImage = mbf.make_image<short>();
  m_currentRgbImage = mbf.make_image<Vector4u>();
}

void RelocaliserApplication::run()
{
  boost::optional<RelocalisationExample> currentExample;

  std::cout << "Start training.\n";

  // First of all, train the relocaliser processing each image from the training folder.
  AverageTimer<boost::chrono::milliseconds> trainingTimer("Training Timer");
  std::ifstream train_infile(g_train_path);
  std::string current_train_file;
  while(train_infile >> current_train_file)
  {
    currentExample = file_read_example(current_train_file);
    
    trainingTimer.start_sync();

    prepare_example_images(*currentExample);

    // Now train the relocaliser.
    m_relocaliser->train(m_currentRgbImage.get(),
                         m_currentDepthImage.get(),
                         m_cameraCalibration.intrinsics_d.projectionParamsSimple.all,
                         currentExample->cameraPose);

    // Finally, increment the index.
    m_trainingSequencePathGenerator->increment_index();

    // Stop the timer before the visualization calls.
    trainingTimer.stop_sync();

    // Update UI
    // show_example(*currentExample);
  }

  std::cout << "Training done, processed " << m_trainingSequencePathGenerator->get_index() << " RGB-D image pairs.\n";

  // Now test the relocaliser accumulating the number of successful relocalisations.
  uint32_t successfulExamples = 0;
  AverageTimer<boost::chrono::milliseconds> testingTimer("Testing Timer");
  std::ifstream test_infile(g_test_path);
  std::string current_test_file;
  while(test_infile >> current_test_file)
  {
    currentExample = file_read_example(current_test_file);
    
    testingTimer.start_sync();
    prepare_example_images(*currentExample);

    // Now relocalise.
    std::vector<Relocaliser::Result> relocaliserResults =
        m_relocaliser->relocalise(m_currentRgbImage.get(),
                                  m_currentDepthImage.get(),
                                  m_cameraCalibration.intrinsics_d.projectionParamsSimple.all);

    // This will store the relocalised pose (if we were successful).

    // gather the predicted poses into vector cameraPoses
    std::vector<Matrix4f> cameraPoses;
    Matrix4f inverseCameraPose;
    uint32_t num_output = m_settingsContainer->get_first_value<uint32_t>("ScoreRelocaliser.maxRelocalisationsToOutput");
    for (uint32_t i = 0; i < num_output; i++)
    {
      if(i < relocaliserResults.size())
        inverseCameraPose = relocaliserResults[i].pose.GetInvM();
      else
        inverseCameraPose.setValues(std::numeric_limits<float>::quiet_NaN());
      cameraPoses.push_back(inverseCameraPose);
    }

    // Save the pose if required.
    if(m_saveRelocalisedPoses)
    {
      for (uint32_t i = 0; i < num_output; i++)
      {
        PosePersister::save_pose_on_thread(cameraPoses[i],
                                         m_outputPosesPathGenerator->make_path("pose-%06i-%i.reloc.txt", currentExample->idx, i));
      }
      std::cout << m_outputPosesPathGenerator->make_path("pose-%06i-%i.reloc.txt", currentExample->idx, 0).string().c_str() << std::endl;
    }

    // Check whether the relocalisation succeeded or not by looking at the ground truth pose (using the 7-scenes
    // 5cm/5deg criterion).
    static const float translationMaxError = 0.05f;
    static const float angleMaxError = 5.f * static_cast<float>(M_PI) / 180.f;
    const bool relocalisationSucceeded =
        pose_matches(currentExample->cameraPose.GetInvM(), inverseCameraPose, translationMaxError, angleMaxError);


    // Update stats.
    successfulExamples += relocalisationSucceeded;

    // Increment path generator indices.
    m_testingSequencePathGenerator->increment_index();
    m_outputPosesPathGenerator->increment_index();

    // Stop the timer before the visualization calls.
    testingTimer.stop_sync();

    // Show the example and print whether the relocalisation succeeded or not.
    // show_example(*currentExample, relocalisationSucceeded ? "Relocalisation OK" : "Relocalisation Failed");
  }

  const int testedExamples = m_testingSequencePathGenerator->get_index();
  std::cout << "Testing done.\n\nEvaluated " << testedExamples << " RGBD frames.\n";
  std::cout << successfulExamples
            << " frames were relocalised correctly (<=5cm translational and <=5deg angular error).\n";
  const float accuracy =
      100.f * (testedExamples > 0 ? static_cast<float>(successfulExamples) / static_cast<float>(testedExamples) : 0.0f);
  std::cout << "Overall accuracy: " << accuracy << "%\n";
  std::cout << trainingTimer << '\n';
  std::cout << testingTimer << '\n';
}


void RelocaliserApplication::train()
{
  boost::optional<RelocalisationExample> currentExample;

  std::cout << "Start training.\n";

  // First of all, train the relocaliser processing each image from the training folder.
  AverageTimer<boost::chrono::milliseconds> trainingTimer("Training Timer");
    // Check that training and testing folders exist.
  if(!bf::is_regular_file(g_train_path))
  {
    throw std::invalid_argument("The specified training path does not exist.");
  }
  std::ifstream train_infile(g_train_path);
  std::string current_train_file;
  while(train_infile >> current_train_file)
  {
    std::cout << current_train_file << std::endl;
    currentExample = file_read_example(current_train_file);
    
    trainingTimer.start_sync();

    prepare_example_images(*currentExample);

    // Now train the relocaliser.
    m_relocaliser->train(m_currentRgbImage.get(),
                         m_currentDepthImage.get(),
                         m_cameraCalibration.intrinsics_d.projectionParamsSimple.all,
                         currentExample->cameraPose);

    // Finally, increment the index.
    m_trainingSequencePathGenerator->increment_index();

    // Stop the timer before the visualization calls.
    trainingTimer.stop_sync();

    // Update UI
    // show_example(*currentExample);
  }
  std::cout << trainingTimer << '\n';

  // save the trained relocalizer
  m_relocaliser->save_to_disk(model_save_path + '/' + g_model_name);

  std::cout << "Training done, processed " << m_trainingSequencePathGenerator->get_index() << " RGB-D image pairs.\n";
  std::cout << "Model saved at " << model_save_path + '/' + g_model_name << std::endl;
}


void RelocaliserApplication::test()
{
  if(!bf::is_directory(model_save_path + '/' + g_model_name))
  {
    throw std::invalid_argument("Pretrained model does not exist.");
  }
  m_relocaliser->load_from_disk(model_save_path + '/' + g_model_name);
  std::cout << "Loading pretrained model from " <<   model_save_path + '/' + g_model_name << std::endl;

  boost::optional<RelocalisationExample> currentExample;

  // Now test the relocaliser accumulating the number of successful relocalisations.
  uint32_t successfulExamples = 0;
  AverageTimer<boost::chrono::milliseconds> testingTimer("Testing Timer");
  if(!bf::is_regular_file(g_test_path))
  {
    throw std::invalid_argument("The specified testing path does not exist.");
  }
  std::ifstream test_infile(g_test_path);
  std::string current_test_file;
  while(test_infile >> current_test_file)
  {
    currentExample = file_read_example(current_test_file);
    
    testingTimer.start_sync();
    prepare_example_images(*currentExample);

    // Now relocalise.
    std::vector<Relocaliser::Result> relocaliserResults =
        m_relocaliser->relocalise(m_currentRgbImage.get(),
                                  m_currentDepthImage.get(),
                                  m_cameraCalibration.intrinsics_d.projectionParamsSimple.all);

    // This will store the relocalised pose (if we were successful).

    // gather the predicted poses into vector cameraPoses
    std::vector<Matrix4f> cameraPoses;
    Matrix4f inverseCameraPose;
    uint32_t num_output = m_settingsContainer->get_first_value<uint32_t>("ScoreRelocaliser.maxRelocalisationsToOutput");
    for (uint32_t i = 0; i < num_output; i++)
    {
      if(i < relocaliserResults.size())
        inverseCameraPose = relocaliserResults[i].pose.GetInvM();
      else
        inverseCameraPose.setValues(std::numeric_limits<float>::quiet_NaN());
      cameraPoses.push_back(inverseCameraPose);
    }

    // Save the pose if required.
    if(m_saveRelocalisedPoses)
    {
      for (uint32_t i = 0; i < num_output; i++)
      {
        PosePersister::save_pose_on_thread(cameraPoses[i],
                                        m_outputPosesPathGenerator->make_path("pose-%06i-%i.reloc.txt", currentExample->idx, i));
        if(i == 0)
        {
          std::cout << m_outputPosesPathGenerator->make_path("pose-%06i-%i.reloc.txt", currentExample->idx, i).string().c_str() << std::endl;
        }
      }
    }

    // Check whether the relocalisation succeeded or not by looking at the ground truth pose (using the 7-scenes
    // 5cm/5deg criterion).
    static const float translationMaxError = 0.05f;
    static const float angleMaxError = 5.f * static_cast<float>(M_PI) / 180.f;
    const bool relocalisationSucceeded =
        pose_matches(currentExample->cameraPose.GetInvM(), inverseCameraPose, translationMaxError, angleMaxError);


    // Update stats.
    successfulExamples += relocalisationSucceeded;
    
    // Increment path generator indices.
    m_testingSequencePathGenerator->increment_index();
    m_outputPosesPathGenerator->increment_index();

    // Stop the timer before the visualization calls.
    testingTimer.stop_sync();

    // Show the example and print whether the relocalisation succeeded or not.
    // show_example(*currentExample, relocalisationSucceeded ? "Relocalisation OK" : "Relocalisation Failed");
  }

  const int testedExamples = m_testingSequencePathGenerator->get_index();
  std::cout << "Testing done.\n\nEvaluated " << testedExamples << " RGBD frames.\n";
  std::cout << successfulExamples
            << " frames were relocalised correctly (<=5cm translational and <=5deg angular error).\n";
  const float accuracy =
      100.f * (testedExamples > 0 ? static_cast<float>(successfulExamples) / static_cast<float>(testedExamples) : 0.0f);
  std::cout << "Overall accuracy: " << accuracy << "%\n";
  std::cout << testingTimer << '\n';
}


void RelocaliserApplication::test4rl()
{
  if(!bf::is_directory(model_save_path + '/' + g_model_name))
  {
    throw std::invalid_argument("Pretrained model does not exist.");
  }
  m_relocaliser->load_from_disk(model_save_path + '/' + g_model_name);
  std::cout << "Loading pretrained model from " <<   model_save_path + '/' + g_model_name << std::endl;

  boost::optional<RelocalisationExample> currentExample;
  std::string g_test_path_string = g_test_path;
  std::string ready_file = g_test_path_string;
  ready_file.replace(ready_file.end() - 9, ready_file.end(), "ready.txt");

  while(1)
  {
    // See if there exsits the ready file indicating the testing data file is ready.
    // if(!bf::is_regular_file(g_test_path))
    if(!bf::is_regular_file(ready_file))
    {
      sleep(0.01);
    }
    else
    {
      // Testing file exsits, now relocalize
      // Now test the relocaliser accumulating the number of successful relocalisations.
      std::ifstream test_infile(g_test_path);
      std::string current_test_file;
      while(test_infile >> current_test_file)
      {
        currentExample = file_read_example(current_test_file);
        
        prepare_example_images(*currentExample);

        // Now relocalise.
        std::vector<Relocaliser::Result> relocaliserResults =
            m_relocaliser->relocalise(m_currentRgbImage.get(),
                                      m_currentDepthImage.get(),
                                      m_cameraCalibration.intrinsics_d.projectionParamsSimple.all);

        // gather the predicted poses into vector cameraPoses
        std::vector<Matrix4f> cameraPoses;
        Matrix4f inverseCameraPose;
        uint32_t num_output = m_settingsContainer->get_first_value<uint32_t>("ScoreRelocaliser.maxRelocalisationsToOutput");
        for (uint32_t i = 0; i < num_output; i++)
        {
          if(i < relocaliserResults.size())
            inverseCameraPose = relocaliserResults[i].pose.GetInvM();
          else
            inverseCameraPose.setValues(std::numeric_limits<float>::quiet_NaN());
          cameraPoses.push_back(inverseCameraPose);
        }

        // Save the pose.
        for (uint32_t i = 0; i < num_output; i++)
        {
          std::string save_pose_path = g_test_path_string;
          save_pose_path.replace(save_pose_path.end() - 9, save_pose_path.end(), "pose-" + std::to_string(i) + ".txt");
          std::ofstream pose_file(save_pose_path);
          pose_file << cameraPoses[i] << std::endl;
          pose_file.close();
          // PosePersister::save_pose_on_thread(cameraPoses[i], save_pose_path);
        }
      }
      remove(g_test_path);
      remove(ready_file.c_str());
    }
  }
}

bool equal(orx::Relocaliser::CorrectPointType a, orx::Relocaliser::CorrectPointType b)
{
  if (std::abs(a[0] - b[0]) < 1e-4 && std::abs(a[1] - b[1]) < 1e-4 && std::abs(a[2] - b[2]) < 1e-4) return true;
  return false;
}

bool lower(orx::Relocaliser::CorrectPointType a, orx::Relocaliser::CorrectPointType b)
{
  if (equal(a, b)) return false; 
  if (std::abs(a[0] - b[0]) > 1e-4)
  {
    if (a[0] < b[0])
    {
      return true;
    }
    else
    {
      return false;
    }
  }
  if (std::abs(a[1] - b[1]) > 1e-4)
  {
    if (a[1] < b[1])
    {
      return true;
    }
    else
    {
      return false;
    }
  }
  if (std::abs(a[2] - b[2]) > 1e-4)
  {
    if (a[2] < b[2])
    {
      return true;
    }
    else
    {
      return false;
    }
  }
  return false; 
}

void qsort(orx::Relocaliser::CorrectPointCloud& pointCloud, int head, int tail)
{
  if (tail <= head) return;

  const orx::Relocaliser::CorrectPointType midPoint = pointCloud[(head + tail) / 2];
  int i = head;
  int j = tail;


  while (i <= j)
  {
    while (lower(pointCloud[i], midPoint)) i++;
    while (lower(midPoint, pointCloud[j])) j--;

    if (i <= j)
    {
      orx::Relocaliser::CorrectPointType tmp = pointCloud[i];
      pointCloud[i] = pointCloud[j];
      pointCloud[j] = tmp;
      i++;
      j--; 
    }
  }

  if (i < tail) qsort(pointCloud, i, tail);
  if (head < j) qsort(pointCloud, head, j);
}

void RelocaliserApplication::test4pcd()
{
  if(!bf::is_directory(model_save_path + '/' + g_model_name))
  {
    throw std::invalid_argument("Pretrained model does not exist.");
  }
  m_relocaliser->load_from_disk(model_save_path + '/' + g_model_name);
  std::cout << "Loading pretrained model from " <<   model_save_path + '/' + g_model_name << std::endl;

  boost::optional<RelocalisationExample> currentExample;

  if(!bf::is_regular_file(g_test_path))
  {
    throw std::invalid_argument("The specified testing path does not exist.");
  }
  std::ifstream test_infile(g_test_path);
  std::string current_test_file;
  orx::Relocaliser::CorrectPointCloud pointCloud;
  int exampleNum = 0;
  while(test_infile >> current_test_file)
  {
    currentExample = file_read_example(current_test_file);
    
    prepare_example_images(*currentExample);

    m_relocaliser->test4pcd(m_currentRgbImage.get(),
                            m_currentDepthImage.get(),
                            m_cameraCalibration.intrinsics_d.projectionParamsSimple.all,
                            currentExample->cameraPose,
			    pointCloud);
  }
  std::cout << "correct points:\t" << pointCloud.size() << std::endl;

  // Sort the point cloud.
  std::cout << "Begin to sort the point cloud!" << std::endl;
  qsort(pointCloud, 0, pointCloud.size() - 1);

  // Merge the same points.
  std::cout << "Merge the same points!" << std::endl;

  orx::Relocaliser::CorrectPointCloud outputPointCloud;

  for (int i = 0; i < pointCloud.size();)
  { 
    int j = i + 1;
    while (j < pointCloud.size() && equal(pointCloud[i], pointCloud[j]))
    {
      for (int k = 3; k < 25; k++)
      {
        pointCloud[i][k] += pointCloud[j][k];
      }
      j++;
    } 
    orx::Relocaliser::CorrectPointType point(14, 0);
    for (int k = 0; k < 3; k++)
    {
      point[k] = pointCloud[i][k];
    }
    for (int k = 3; k < 13; k++)
    {
      if (pointCloud[i][k] + pointCloud[i][k + 10] > 0)
      {
        point[k] = pointCloud[i][k] / (pointCloud[i][k] + pointCloud[i][k + 10]);
      }
      else
      {
        point[k] = -1;
      }
    }
    if (pointCloud[i][23] + pointCloud[i][24] > 0)
    {
      point[13] = pointCloud[i][23] / (pointCloud[i][23] + pointCloud[i][24]);
    }
    else
    {
      point[13] = -1;
    }
    outputPointCloud.push_back(point);
    i = j;
  }

  std::cout << "Final number of points:\t" << outputPointCloud.size() << std::endl;

  // Output the results.
  std::cout << "Start to output the results!" << std::endl;
  std::fstream outputFile(model_save_path + "/pcd_correspondence.txt", std::ios_base::out);
  for (int i = 0; i < outputPointCloud.size(); i++)
  {
    for (int j = 0; j < 13; j++)
    {
      outputFile << outputPointCloud[i][j] << " ";
    }
    outputFile << outputPointCloud[i][13] << std::endl; 
  }
  outputFile.close();
}
//#################### PRIVATE MEMBER FUNCTIONS ####################

boost::optional<RelocaliserApplication::RelocalisationExample>
    RelocaliserApplication::rel_read_example(bool is_trainning_data) const // true for train
{
  bf::path currentDepthPath;
  bf::path currentRgbPath;
  bf::path currentPosePath;

  if (is_trainning_data) // train
  {
    char c_depth[200];
    //std::sprintf(c_depth, "%s/frame-%06d.depth.png", g_train_path, g_train_id);
    std::sprintf(c_depth, "%s/%06d_depth.png", g_train_path, g_train_id);
    currentDepthPath = c_depth;
    char c_RGB[200];
    //std::sprintf(c_RGB, "%s/frame-%06d.color.png", g_train_path, g_train_id);
    std::sprintf(c_RGB, "%s/%06d_color.png", g_train_path, g_train_id);
    currentRgbPath = c_RGB;
    char c_pose[200];
    //std::sprintf(c_pose, "%s/frame-%06d.pose.txt", g_train_path, g_train_id);
    std::sprintf(c_pose, "%s/%06d_pose.txt", g_train_path, g_train_id);
    currentPosePath = c_pose;
    g_train_id += g_train_step;
  }
  else // test
  {
    char c_depth[200];
    //std::sprintf(c_depth, "%s/frame-%06d.depth.png", g_test_path, g_test_id);
    std::sprintf(c_depth, "%s/%06d_depth.png", g_test_path, g_test_id);
    currentDepthPath = c_depth;
    char c_RGB[200];
    //std::sprintf(c_RGB, "%s/frame-%06d.color.png", g_test_path, g_test_id);
    std::sprintf(c_RGB, "%s/%06d_color.png", g_test_path, g_test_id);
    currentRgbPath = c_RGB;
    char c_pose[200];
    //std::sprintf(c_pose, "%s/frame-%06d.pose.txt", g_test_path, g_test_id);
    std::sprintf(c_pose, "%s/%06d_pose.txt", g_test_path, g_test_id);
    currentPosePath = c_pose;
    g_test_id += g_test_step;
  }
  std::cout << "currentDepthPath " << currentDepthPath.string().c_str() << std::endl; 

  if(bf::is_regular(currentDepthPath) && bf::is_regular(currentRgbPath) && bf::is_regular(currentPosePath))
  {
    RelocalisationExample example;

    // Read the images.
    example.depthImage = cv::imread(currentDepthPath.string().c_str(), cv::IMREAD_ANYDEPTH);
    example.rgbImage = cv::imread(currentRgbPath.string().c_str());

    // The files store the inverse camera pose.
    example.cameraPose.SetInvM(read_pose_from_file(currentPosePath));

    // Convert from BGR to RGBA.
    cv::cvtColor(example.rgbImage, example.rgbImage, CV_BGR2RGBA);

    return example;
  }
  return boost::none;
}


boost::optional<RelocaliserApplication::RelocalisationExample>
    RelocaliserApplication::file_read_example(std::string file_name) const // true for train
{
  bf::path currentDepthPath;
  bf::path currentRgbPath;
  bf::path currentPosePath;
  uint32_t idx;

  currentDepthPath = file_name.c_str();
  // idx = std::stoi(file_name.substr(file_name.length() - 16, 6));
  std::string basename = currentDepthPath.filename().string();
  idx = std::stoi(basename.replace(basename.end() - 10, basename.end(), ""));

  currentRgbPath = file_name.replace(file_name.end() - 10, file_name.end(), "_color.png");
  currentPosePath = file_name.replace(file_name.end() - 10, file_name.end(), "_pose.txt");
  


  if(bf::is_regular(currentDepthPath) && bf::is_regular(currentRgbPath) && bf::is_regular(currentPosePath))
  {
    RelocalisationExample example;

    example.idx = idx;
    // Read the images.
    example.depthImage = cv::imread(currentDepthPath.string().c_str(), cv::IMREAD_ANYDEPTH);
    example.rgbImage = cv::imread(currentRgbPath.string().c_str());

    // The files store the inverse camera pose.
    example.cameraPose.SetInvM(read_pose_from_file(currentPosePath));

    // Convert from BGR to RGBA.
    cv::cvtColor(example.rgbImage, example.rgbImage, CV_BGR2RGBA);

    return example;
  }
  throw std::invalid_argument("Couldn't read files.");
}

boost::optional<RelocaliserApplication::RelocalisationExample>
    RelocaliserApplication::read_example(const RelocaliserApplication::SequentialPathGenerator_Ptr &pathGenerator) const
{
  bf::path currentDepthPath = pathGenerator->make_path(m_depthImageMask);
  bf::path currentRgbPath = pathGenerator->make_path(m_rgbImageMask);
  bf::path currentPosePath = pathGenerator->make_path(m_poseFileMask);

  if(bf::is_regular(currentDepthPath) && bf::is_regular(currentRgbPath) && bf::is_regular(currentPosePath))
  {
    RelocalisationExample example;

    // Read the images.
    example.depthImage = cv::imread(currentDepthPath.string().c_str(), cv::IMREAD_ANYDEPTH);
    example.rgbImage = cv::imread(currentRgbPath.string().c_str());

    // The files store the inverse camera pose.
    example.cameraPose.SetInvM(read_pose_from_file(currentPosePath));

    // Convert from BGR to RGBA.
    cv::cvtColor(example.rgbImage, example.rgbImage, CV_BGR2RGBA);

    return example;
  }

  return boost::none;
}

void RelocaliserApplication::prepare_example_images(const RelocaliserApplication::RelocalisationExample &currentExample)
{
  const Vector2i depthImageDims(currentExample.depthImage.cols, currentExample.depthImage.rows);
  const Vector2i rgbImageDims(currentExample.rgbImage.cols, currentExample.rgbImage.rows);

  // Copy the Mats into the ITM images (the depth image needs conversion to float according to the calibration).

  // Resize them (usually NOOP).
  m_currentRawDepthImage->ChangeDims(depthImageDims);
  m_currentDepthImage->ChangeDims(depthImageDims);
  m_currentRgbImage->ChangeDims(rgbImageDims);

  // Perform copy using a Mat wrapper.
  currentExample.depthImage.convertTo(
      cv::Mat(depthImageDims.y, depthImageDims.x, CV_16SC1, m_currentRawDepthImage->GetData(MEMORYDEVICE_CPU)), CV_16S);
  currentExample.rgbImage.copyTo(
      cv::Mat(rgbImageDims.y, rgbImageDims.x, CV_8UC4, m_currentRgbImage->GetData(MEMORYDEVICE_CPU)));

  // Update them on the device.
  m_currentRawDepthImage->UpdateDeviceFromHost();
  m_currentRgbImage->UpdateDeviceFromHost();

  // Use the viewBuilder to prepare the depth image.
  m_viewBuilder->ConvertDepthAffineToFloat(
      m_currentDepthImage.get(), m_currentRawDepthImage.get(), m_cameraCalibration.disparityCalib.GetParams());
}

void RelocaliserApplication::show_example(const RelocalisationExample &example, const std::string &uiText) const
{
  static const std::string WINDOW_NAME = "Relocalisation GUI";

  // Setup a named window.
  cv::namedWindow(WINDOW_NAME, CV_WINDOW_AUTOSIZE);

  // Allocate a canvas big enough to hold the colour and depth image side by side.
  cv::Mat canvas = cv::Mat::zeros(std::max(example.depthImage.rows, example.rgbImage.rows),
                                  example.depthImage.cols + example.rgbImage.cols,
                                  CV_8UC3);

  // Copy the colour image to its location on the canvas (converting it into BGR format since that's what OpenCV uses
  // for visualization).
  cv::cvtColor(example.rgbImage, canvas(cv::Rect(0, 0, example.rgbImage.cols, example.rgbImage.rows)), CV_RGBA2BGR);

  // Normalize the depth image (black is very close, white is far away) and copy it to its location on the canvas.
  cv::Mat processedDepth;
  cv::normalize(example.depthImage, processedDepth, 0, 255, cv::NORM_MINMAX, CV_8U);
  cv::cvtColor(processedDepth,
               canvas(cv::Rect(example.rgbImage.cols, 0, example.depthImage.cols, example.depthImage.rows)),
               CV_GRAY2BGR);

  // Draw the text in the top-left corner, if required.
  if(!uiText.empty())
  {
    const double fontSize = 1.5;
    const int thickness = 2;

    // Compute the text's bounding box (we actually only care about its height).
    int baseLine = 0;
    cv::Size textSize = cv::getTextSize(uiText, cv::FONT_HERSHEY_SIMPLEX, fontSize, thickness, &baseLine);

    // Write the text on the image applying a "Poor man's shadow effect".
    cv::putText(canvas,
                uiText,
                cv::Point(12, 12 + textSize.height),
                cv::FONT_HERSHEY_SIMPLEX,
                fontSize,
                cv::Scalar::all(0),
                thickness);
    cv::putText(canvas,
                uiText,
                cv::Point(10, 10 + textSize.height),
                cv::FONT_HERSHEY_SIMPLEX,
                fontSize,
                cv::Scalar::all(255),
                thickness);
  }

  // Actualy show the image.
  cv::imshow(WINDOW_NAME, canvas);
  cv::waitKey(1);
}
}
