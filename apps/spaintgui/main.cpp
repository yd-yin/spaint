/**
 * spaintgui: main.cpp
 * Copyright (c) Torr Vision Group, University of Oxford, 2015. All rights reserved.
 */

#include <cstdlib>
#include <iostream>
#include <string>

#include <boost/program_options.hpp>
#include <boost/tokenizer.hpp>

// Note: This must appear before anything that could include SDL.h, since it includes boost/asio.hpp, a header that has a WinSock conflict with SDL.h.
#include "Application.h"

#ifdef WITH_ARRAYFIRE
  #include <arrayfire.h>
#endif

#include <InputSource/OpenNIEngine.h>
#ifdef WITH_REALSENSE
#include <InputSource/RealSenseEngine.h>
#endif

#ifdef WITH_GLUT
#include <spaint/ogl/WrappedGLUT.h>
#endif

#ifdef WITH_OVR
#include <OVR_CAPI.h>
#endif

#ifdef WITH_OPENCV
#include <spaint/fiducials/ArUcoFiducialDetector.h>
#endif

#include <itmx/base/MemoryBlockFactory.h>

#include <spaint/imagesources/AsyncImageSourceEngine.h>

#include <tvgutil/filesystem/PathFinder.h>
#include <tvgutil/misc/SettingsContainer.h>

#include "core/ObjectivePipeline.h"
#include "core/SemanticPipeline.h"
#include "core/SLAMPipeline.h"

using namespace InputSource;
using namespace ITMLib;

using namespace itmx;
using namespace spaint;
using namespace tvgutil;

//#################### NAMESPACE ALIASES ####################

namespace bf = boost::filesystem;
namespace po = boost::program_options;

//#################### TYPES ####################

struct CommandLineArguments
{
  // User-specifiable arguments
  bool batch;
  std::string calibrationFilename;
  bool cameraAfterDisk;
  std::vector<std::string> depthImageMasks;
  bool detectFiducials;
  std::string experimentTag;
  int initialFrameNumber;
  std::string leapFiducialID;
  bool mapSurfels;
  bool noRelocaliser;
  bool noTracker;
  std::string openNIDeviceURI;
  std::string pipelineType;
  std::vector<std::string> poseFileMasks;
  size_t prefetchBufferCapacity;
  bool renderFiducials;
  std::vector<std::string> rgbImageMasks;
  bool saveMeshOnExit;
  std::vector<std::string> sequenceSpecifiers;
  std::vector<std::string> sequenceTypes;
  std::vector<std::string> trackerSpecifiers;
  bool trackObject;
  bool trackSurfels;

  // Derived arguments
  std::vector<bf::path> sequenceDirs;
};

//#################### FUNCTIONS ####################

/**
 * \brief Checks whether or not the specified camera subengine is able to provide depth images.
 *
 * \note If the check fails, the camera subengine will be deallocated.
 *
 * \param cameraSubengine The camera subengine to check.
 * \return                The camera subengine, if it is able to provide depth images, or NULL otherwise.
 */
ImageSourceEngine *check_camera_subengine(ImageSourceEngine *cameraSubengine)
{
  if(cameraSubengine->getDepthImageSize().x == 0)
  {
    delete cameraSubengine;
    return NULL;
  }
  else return cameraSubengine;
}

/**
 * \brief Attempts to make a camera subengine to read images from any suitable camera that is attached.
 *
 * \param args  The program's command-line arguments.
 * \return      The camera subengine, if a suitable camera is attached, or NULL otherwise.
 */
ImageSourceEngine *make_camera_subengine(const CommandLineArguments& args)
{
  ImageSourceEngine *cameraSubengine = NULL;

#ifdef WITH_OPENNI
  // Probe for an OpenNI camera.
  if(cameraSubengine == NULL)
  {
    std::cout << "[spaint] Probing OpenNI camera: " << args.openNIDeviceURI << '\n';
    boost::optional<std::string> uri = args.openNIDeviceURI == "Default" ? boost::none : boost::optional<std::string>(args.openNIDeviceURI);
    bool useInternalCalibration = !uri; // if reading from a file, assume that the provided calibration is to be used
    cameraSubengine = check_camera_subengine(new OpenNIEngine(args.calibrationFilename.c_str(), uri ? uri->c_str() : NULL, useInternalCalibration
#if USE_LOW_USB_BANDWIDTH_MODE
      // If there is insufficient USB bandwidth available to support 640x480 RGB input, use 320x240 instead.
      , Vector2i(320, 240)
#endif
    ));
  }
#endif

#ifdef WITH_REALSENSE
  // Probe for a RealSense camera.
  if(cameraSubengine == NULL)
  {
    std::cout << "[spaint] Probing RealSense camera\n";
    cameraSubengine = check_camera_subengine(new RealSenseEngine(args.calibrationFilename.c_str()));
  }
#endif

  return cameraSubengine;
}

/**
 * \brief Makes the overall tracker configuration based on any tracker specifiers that were passed in on the command line.
 *
 * \param args  The program's command-line arguments.
 * \return      The overall tracker configuration.
 */
std::string make_tracker_config(CommandLineArguments& args)
{
  std::string result;

  // Determine the number of different trackers that will be needed.
  size_t trackerCount = args.sequenceSpecifiers.size();
  if(trackerCount == 0 || args.cameraAfterDisk) ++trackerCount;

  // If more than one tracker is needed, make the overall tracker a composite.
  if(trackerCount > 1) result += "<tracker type='composite' policy='sequential'>";

  // For each tracker that is needed:
  for(size_t i = 0; i < trackerCount; ++i)
  {
    // Look to see if the user specified an explicit tracker specifier for it on the command line; if not, use a default tracker specifier.
    const std::string trackerSpecifier = i < args.trackerSpecifiers.size() ? args.trackerSpecifiers[i] : "InfiniTAM";

    // Separate the tracker specifier into chunks.
    typedef boost::char_separator<char> sep;
    typedef boost::tokenizer<sep> tokenizer;

    tokenizer tok(trackerSpecifier.begin(), trackerSpecifier.end(), sep("+"));
    std::vector<std::string> chunks(tok.begin(), tok.end());

    // Add a tracker configuration based on the specifier chunks to the overall tracker configuration.
    // If more than one chunk is involved, bundle the subsidiary trackers into a refining composite.
    size_t chunkCount = chunks.size();
    if(chunkCount > 1) result += "<tracker type='composite'>";

    for(size_t i = 0; i < chunkCount; ++i)
    {
      if(chunks[i] == "InfiniTAM")
      {
        result += "<tracker type='infinitam'/>";
      }
      else if(chunks[i] == "Disk")
      {
        if(args.poseFileMasks.size() < i)
        {
          // If this happens it's because at least one mask was specified with the -p flag,
          // otherwise postprocess_arguments would have taken care of supplying the default masks.
          throw std::invalid_argument("Not enough pose file masks have been specified with the -p flag.");
        }

        result += "<tracker type='infinitam'><params>type=file,mask=" + args.poseFileMasks[i] + "</params></tracker>";
      }
      else
      {
        result += "<tracker type='import'><params>builtin:" + chunks[i] + "</params></tracker>";
      }
    }

    // If more than one chunk was involved, add the necessary closing tag for the refining composite.
    if(chunkCount > 1) result += "</tracker>";
  }

  // If more than one tracker was needed, add the necessary closing tag for the overall composite.
  if(trackerCount > 1) result += "</tracker>";

  return result;
}

/**
 * \brief Post-process the program's command-line arguments.
 *
 * \param args  The program's command-line arguments.
 * \return      true, if the program should continue after post-processing its arguments, or false otherwise.
 */
bool postprocess_arguments(CommandLineArguments& args)
{
  // If the user specifies both sequence and explicit depth / RGB image / pose mask flags, print an error message.
  if(!args.sequenceSpecifiers.empty() && (!args.depthImageMasks.empty() || !args.poseFileMasks.empty() || !args.rgbImageMasks.empty()))
  {
    std::cout << "Error: Either sequence flags or explicit depth / RGB image / pose mask flags may be specified, but not both.\n";
    return false;
  }

  // For each sequence (if any) that the user specifies (either via a sequence name or a path), set the depth / RGB image masks appropriately.
  for(size_t i = 0, size = args.sequenceSpecifiers.size(); i < size; ++i)
  {
    // Determine the sequence type.
    const std::string sequenceType = i < args.sequenceTypes.size() ? args.sequenceTypes[i] : "sequence";

    // Determine the directory containing the sequence and record it for later use.
    const std::string& sequenceSpecifier = args.sequenceSpecifiers[i];
    const bf::path dir = bf::is_directory(sequenceSpecifier)
      ? sequenceSpecifier
      : find_subdir_from_executable(sequenceType + "s") / sequenceSpecifier;
    args.sequenceDirs.push_back(dir);

    // Set the depth / RGB image masks.
    args.depthImageMasks.push_back((dir / "depthm%06i.pgm").string());
    args.poseFileMasks.push_back((dir / "posem%06i.txt").string());
    args.rgbImageMasks.push_back((dir / "rgbm%06i.ppm").string());
  }

  // If the user hasn't explicitly specified a calibration file, try to find one in the first sequence directory (if it exists).
  if(args.calibrationFilename == "" && !args.sequenceDirs.empty())
  {
    bf::path defaultCalibrationFilename = args.sequenceDirs[0] / "calib.txt";
    if(bf::exists(defaultCalibrationFilename))
    {
      args.calibrationFilename = defaultCalibrationFilename.string();
    }
  }

  // If the user wants to enable surfel tracking, make sure that surfel mapping is also enabled.
  if(args.trackSurfels) args.mapSurfels = true;

  // If the user wants to enable fiducial rendering or specifies a fiducial to use for the Leap Motion,
  // make sure that fiducial detection is enabled.
  if(args.renderFiducials || args.leapFiducialID != "")
  {
    args.detectFiducials = true;
  }

  return true;
}

/**
 * \brief Sets the scene parameters from GlobalParameters, allowing the user to specify ad hoc values for voxel size, truncation distance, etc...
 *
 * \param sceneParams The scene parameters to modify.
 */
void set_scene_params_from_global_options(const SettingsContainer_CPtr &settings, ITMSceneParams &sceneParams)
{
#define GET_PARAM(type, name, defaultValue) sceneParams.name = settings->get_first_value<type>("SceneParams."#name, defaultValue)

  // Use the default values from InfiniTAM.
  GET_PARAM(int, maxW, 100);
  GET_PARAM(float, mu, 0.02f);
  GET_PARAM(bool, stopIntegratingAtMaxW, false);
  GET_PARAM(float, viewFrustum_max, 3.0f);
  GET_PARAM(float, viewFrustum_min, 0.2f);
  GET_PARAM(float, voxelSize, 0.005f);

#undef GET_PARAM
}

/**
 * \brief Sets the surfel scene parameters from GlobalParameters, allowing the user to specify ad hoc values for surfel radius, etc...
 *
 * \param surfelSceneParams The surfel scene parameters to modify.
 */
void set_surfel_scene_params_from_global_options(const SettingsContainer_CPtr &settings, ITMSurfelSceneParams &surfelSceneParams)
{
#define GET_PARAM(type, name, defaultValue) surfelSceneParams.name = settings->get_first_value<type>("SurfelSceneParams."#name, defaultValue)

  // Use the default values from InfiniTAM.
  GET_PARAM(float, deltaRadius, 0.5f);
  GET_PARAM(float, gaussianConfidenceSigma, 0.6f);
  GET_PARAM(float, maxMergeAngle, static_cast<float>(20 * M_PI / 180));
  GET_PARAM(float, maxMergeDist, 0.01f);
  GET_PARAM(float, maxSurfelRadius, 0.004f);
  GET_PARAM(float, minRadiusOverlapFactor, 3.5f);
  GET_PARAM(float, stableSurfelConfidence, 25.0f);
  GET_PARAM(int, supersamplingFactor, 4);
  GET_PARAM(float, trackingSurfelMaxDepth, 1.0f);
  GET_PARAM(float, trackingSurfelMinConfidence, 5.0f);
  GET_PARAM(int, unstableSurfelPeriod, 20);
  GET_PARAM(int, unstableSurfelZOffset, 10000000);
  GET_PARAM(bool, useGaussianSampleConfidence, true);
  GET_PARAM(bool, useSurfelMerging, true);

#undef GET_PARAM
}

/**
 * \brief Stores the parsed options in a SettingsContainer instance.
 *
 * \param parsedOptions The options to store in the SettingsContainer.
 * \param settings      The settings container.
 */
void store_parsed_options_into_settings(const po::parsed_options& parsedOptions, const SettingsContainer_Ptr &settings)
{
  for(size_t optionIdx = 0; optionIdx < parsedOptions.options.size(); ++optionIdx)
  {
    const po::basic_option<char> &option = parsedOptions.options[optionIdx];

    // Add all values in the correct order.
    for(size_t valueIdx = 0; valueIdx < option.value.size(); ++valueIdx)
    {
      settings->add_value(option.string_key, option.value[valueIdx]);
    }
  }
}

/**
 * \brief Parse any command-line arguments passed in by the user.
 *
 * \param argc  The command-line argument count.
 * \param argv  The raw command-line arguments.
 * \param args  The parsed command-line arguments.
 * \return      true, if the program should continue after parsing the command-line arguments, or false otherwise.
 */
bool parse_command_line(int argc, char *argv[], CommandLineArguments& args, const SettingsContainer_Ptr &settings)
{
  // Specify the possible options.
  po::options_description genericOptions("Generic options");
  genericOptions.add_options()
    ("help", "produce help message")
    ("batch", po::bool_switch(&args.batch), "don't wait for user input before starting the reconstruction and terminate immediately")
    ("calib,c", po::value<std::string>(&args.calibrationFilename)->default_value(""), "calibration filename")
    ("cameraAfterDisk", po::bool_switch(&args.cameraAfterDisk), "switch to the camera after a disk sequence")
    ("configFile,f", po::value<std::string>(), "additional parameters filename")
    ("detectFiducials", po::bool_switch(&args.detectFiducials), "enable fiducial detection")
    ("experimentTag", po::value<std::string>(&args.experimentTag)->default_value(""), "experiment tag")
    ("leapFiducialID", po::value<std::string>(&args.leapFiducialID)->default_value(""), "the ID of the fiducial to use for the Leap Motion")
    ("mapSurfels", po::bool_switch(&args.mapSurfels), "enable surfel mapping")
    ("noRelocaliser", po::bool_switch(&args.noRelocaliser), "don't use the relocaliser")
    ("noTracker", po::bool_switch(&args.noTracker), "don't use any tracker")
    ("pipelineType", po::value<std::string>(&args.pipelineType)->default_value("semantic"), "pipeline type")
    ("renderFiducials", po::bool_switch(&args.renderFiducials), "enable fiducial rendering")
    ("saveMeshOnExit", po::bool_switch(&args.saveMeshOnExit), "save reconstructed mesh on exit")
    ("trackerSpecifier,t", po::value<std::vector<std::string> >(&args.trackerSpecifiers)->multitoken(), "tracker specifier")
    ("trackSurfels", po::bool_switch(&args.trackSurfels), "enable surfel mapping and tracking")
  ;

  po::options_description cameraOptions("Camera options");
  cameraOptions.add_options()
    ("uri,u", po::value<std::string>(&args.openNIDeviceURI)->default_value("Default"), "OpenNI device URI")
  ;

  po::options_description diskSequenceOptions("Disk sequence options");
  diskSequenceOptions.add_options()
    ("depthMask,d", po::value<std::vector<std::string> >(&args.depthImageMasks)->multitoken(), "depth image mask")
    ("initialFrame,n", po::value<int>(&args.initialFrameNumber)->default_value(0), "initial frame number")
    ("poseMask,p", po::value<std::vector<std::string> >(&args.poseFileMasks)->multitoken(), "pose file mask")
    ("prefetchBufferCapacity,b", po::value<size_t>(&args.prefetchBufferCapacity)->default_value(60), "capacity of the prefetch buffer")
    ("rgbMask,r", po::value<std::vector<std::string> >(&args.rgbImageMasks)->multitoken(), "RGB image mask")
    ("sequenceSpecifier,s", po::value<std::vector<std::string> >(&args.sequenceSpecifiers)->multitoken(), "sequence specifier")
    ("sequenceType", po::value<std::vector<std::string> >(&args.sequenceTypes)->multitoken(), "sequence type")
  ;

  po::options_description objectivePipelineOptions("Objective pipeline options");
  objectivePipelineOptions.add_options()
    ("trackObject", po::bool_switch(&args.trackObject), "track the object")
  ;

  po::options_description options;
  options.add(genericOptions);
  options.add(cameraOptions);
  options.add(diskSequenceOptions);
  options.add(objectivePipelineOptions);

  // Actually parse the command line.
  po::variables_map vm;
  po::parsed_options parsedCommandLineOptions = po::parse_command_line(argc, argv, options);

  // Store the parsed options in both the variables map and the SettingsContainer.
  po::store(parsedCommandLineOptions, vm);
  store_parsed_options_into_settings(parsedCommandLineOptions, settings);

  // Parse options from configuration file, if necessary.
  if(vm.count("configFile"))
  {
    // Allow unregistered options: those are added to the settings container, to be used by other classes.
    po::parsed_options parsedConfigFileOptions = po::parse_config_file<char>(vm["configFile"].as<std::string>().c_str(), options, true);

    // Store registered options in the variable map
    po::store(parsedConfigFileOptions, vm);

    // Store all options (including unregistered ones) into the SettingsContainer.
    store_parsed_options_into_settings(parsedConfigFileOptions, settings);
  }

  po::notify(vm);

  std::cout << "Global settings:\n" << *settings << '\n';

  // If the user specifies the --help flag, print a help message.
  if(vm.count("help"))
  {
    std::cout << options << '\n';
    return false;
  }

  return true;
}

/**
 * \brief Outputs the specified error message and terminates the program with the specified exit code.
 *
 * \param message The error message.
 * \param code    The exit code.
 */
void quit(const std::string& message, int code = EXIT_FAILURE)
{
  std::cerr << message << '\n';
  SDL_Quit();
  exit(code);
}

int main(int argc, char *argv[])
try
{
  // Setup the settings.
  Settings_Ptr settings(new Settings);
  settings->trackerConfig = NULL; // The tracker is handled by the tracker factory.


  // Parse and post-process the command-line arguments.
  CommandLineArguments args;
  if(!parse_command_line(argc, argv, args, settings) || !postprocess_arguments(args))
  {
    return 0;
  }

  // Initialise SDL.
  if(SDL_Init(SDL_INIT_VIDEO) < 0)
  {
    quit("Error: Failed to initialise SDL.");
  }

#ifdef WITH_GLUT
  // Initialise GLUT (used for text rendering only).
  glutInit(&argc, argv);
#endif

#ifdef WITH_OVR
  // If we built with Rift support, initialise the Rift SDK.
  ovr_Initialize();
#endif

  // Set scene parameters from configuration.
  set_scene_params_from_global_options(settings, settings->sceneParams);
  set_surfel_scene_params_from_global_options(settings, settings->surfelSceneParams);

  if(args.cameraAfterDisk || !args.noRelocaliser) settings->behaviourOnFailure = ITMLibSettings::FAILUREMODE_RELOCALISE;

  // Pass the device type to the memory block factory.
  MemoryBlockFactory::instance().set_device_type(settings->deviceType);

  // Construct the image source engine.
  boost::shared_ptr<CompositeImageSourceEngine> imageSourceEngine(new CompositeImageSourceEngine);

  // Add a subengine for each disk sequence specified.
  for(size_t i = 0; i < args.depthImageMasks.size(); ++i)
  {
    const std::string& depthImageMask = args.depthImageMasks[i];
    const std::string& rgbImageMask = args.rgbImageMasks[i];

    std::cout << "[spaint] Reading images from disk: " << rgbImageMask << ' ' << depthImageMask << '\n';
    ImageMaskPathGenerator pathGenerator(rgbImageMask.c_str(), depthImageMask.c_str());
    imageSourceEngine->addSubengine(new AsyncImageSourceEngine(
      new ImageFileReader<ImageMaskPathGenerator>(args.calibrationFilename.c_str(), pathGenerator, args.initialFrameNumber),
      args.prefetchBufferCapacity
    ));
  }

  // If no disk sequences were specified, or we want to switch to the camera once all the disk sequences finish, add a camera subengine.
  if(args.depthImageMasks.empty() || args.cameraAfterDisk)
  {
    ImageSourceEngine *cameraSubengine = make_camera_subengine(args);
    if(cameraSubengine != NULL) imageSourceEngine->addSubengine(cameraSubengine);
  }

  // Construct the fiducial detector (if any).
  FiducialDetector_CPtr fiducialDetector;
#ifdef WITH_OPENCV
  fiducialDetector.reset(new ArUcoFiducialDetector(settings));
#endif

  // Construct the pipeline.
  const size_t maxLabelCount = 10;
  SLAMComponent::MappingMode mappingMode = args.mapSurfels ? SLAMComponent::MAP_BOTH : SLAMComponent::MAP_VOXELS_ONLY;
  SLAMComponent::TrackingMode trackingMode = args.trackSurfels ? SLAMComponent::TRACK_SURFELS : SLAMComponent::TRACK_VOXELS;

  MultiScenePipeline_Ptr pipeline;
  if(args.pipelineType == "semantic")
  {
    const unsigned int seed = 12345;
    pipeline.reset(new SemanticPipeline(
      settings,
      Application::resources_dir().string(),
      maxLabelCount,
      imageSourceEngine,
      seed,
      make_tracker_config(args),
      mappingMode,
      trackingMode,
      fiducialDetector,
      args.detectFiducials
    ));
  }
  else if(args.pipelineType == "objective")
  {
    pipeline.reset(new ObjectivePipeline(
      settings,
      Application::resources_dir().string(),
      maxLabelCount,
      imageSourceEngine,
      make_tracker_config(args),
      mappingMode,
      trackingMode,
      fiducialDetector,
      args.detectFiducials,
      !args.trackObject
    ));
  }
  else if(args.pipelineType == "slam")
  {
    pipeline.reset(new SLAMPipeline(settings,
                                    Application::resources_dir().string(),
                                    imageSourceEngine,
                                    make_tracker_config(args),
                                    mappingMode,
                                    trackingMode));
  }
  else throw std::runtime_error("Unknown pipeline type: " + args.pipelineType);

#ifdef WITH_LEAP
  // Set the ID of the fiducial to use for the Leap Motion (if any).
  pipeline->get_model()->set_leap_fiducial_id(args.leapFiducialID);
#endif

  // Configure the application
  Application app(pipeline, args.renderFiducials);
  app.set_save_mesh_on_exit(args.saveMeshOnExit);
  app.set_batch_mode(args.batch);

  // Run the application.
  bool runSucceeded = app.run();

#ifdef WITH_OVR
  // If we built with Rift support, shut down the Rift SDK.
  ovr_Shutdown();
#endif

  // Shut down SDL.
  SDL_Quit();

  return runSucceeded ? EXIT_SUCCESS : EXIT_FAILURE;
}
catch(std::exception& e)
{
  std::cerr << e.what() << '\n';
  return EXIT_FAILURE;
}
