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

#if defined(WITH_ARRAYFIRE) && defined(WITH_CUDA)
#include <af/cuda.h>
#endif

#include <InputSource/IdleImageSourceEngine.h>
#include <InputSource/OpenNIEngine.h>
#if WITH_LIBROYALE
#include <InputSource/PicoFlexxEngine.h>
#endif
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
#include <itmx/imagesources/AsyncImageSourceEngine.h>
#ifdef WITH_ZED
#include <itmx/imagesources/ZedImageSourceEngine.h>
#endif

#include <tvgutil/filesystem/PathFinder.h>

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
  //~~~~~~~~~~~~~~~~~~~~ PUBLIC VARIABLES ~~~~~~~~~~~~~~~~~~~~

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
  std::string modelSpecifier;
  bool noRelocaliser;
  std::string openNIDeviceURI;
  std::string pipelineType;
  size_t prefetchBufferCapacity;
  std::string relocaliserType;
  bool renderFiducials;
  std::vector<std::string> rgbImageMasks;
  bool saveMeshOnExit;
  bool saveModelsOnExit;
  std::vector<std::string> sequenceSpecifiers;
  std::vector<std::string> sequenceTypes;
  std::string subwindowConfigurationIndex;
  std::vector<std::string> trackerSpecifiers;
  bool trackObject;
  bool trackSurfels;

  // Derived arguments
  boost::optional<bf::path> modelDir;
  std::vector<bf::path> sequenceDirs;

  //~~~~~~~~~~~~~~~~~~~~ PUBLIC MEMBER FUNCTIONS ~~~~~~~~~~~~~~~~~~~~

  /**
   * \brief Adds the command-line arguments to a settings object.
   *
   * \param settings  The settings object.
   */
  void add_to_settings(const Settings_Ptr& settings)
  {
    #define ADD_SETTING(arg) settings->add_value(#arg, boost::lexical_cast<std::string>(arg))
    #define ADD_SETTINGS(arg) for(size_t i = 0; i < arg.size(); ++i) { settings->add_value(#arg, boost::lexical_cast<std::string>(arg[i])); }
      ADD_SETTING(batch);
      ADD_SETTING(calibrationFilename);
      ADD_SETTINGS(depthImageMasks);
      ADD_SETTING(detectFiducials);
      ADD_SETTING(experimentTag);
      ADD_SETTING(initialFrameNumber);
      ADD_SETTING(leapFiducialID);
      ADD_SETTING(mapSurfels);
      ADD_SETTING(modelSpecifier);
      ADD_SETTING(noRelocaliser);
      ADD_SETTING(openNIDeviceURI);
      ADD_SETTING(pipelineType);
      ADD_SETTING(prefetchBufferCapacity);
      ADD_SETTING(relocaliserType);
      ADD_SETTING(renderFiducials);
      ADD_SETTINGS(rgbImageMasks);
      ADD_SETTING(saveMeshOnExit);
      ADD_SETTING(saveModelsOnExit);
      ADD_SETTINGS(sequenceSpecifiers);
      ADD_SETTINGS(sequenceTypes);
      ADD_SETTING(subwindowConfigurationIndex);
      ADD_SETTINGS(trackerSpecifiers);
      ADD_SETTING(trackObject);
      ADD_SETTING(trackSurfels);
    #undef ADD_SETTINGS
    #undef ADD_SETTING
  }
};

//#################### FUNCTIONS ####################

/**
 * \brief Adds any unregistered options in a set of parsed options to a settings object.
 *
 * \param parsedOptions The set of parsed options.
 * \param settings      The settings object.
 */
void add_unregistered_options_to_settings(const po::parsed_options& parsedOptions, const Settings_Ptr& settings)
{
  for(size_t i = 0, optionCount = parsedOptions.options.size(); i < optionCount; ++i)
  {
    const po::basic_option<char>& option = parsedOptions.options[i];
    if(option.unregistered)
    {
      // Add all the specified values for the option in the correct order.
      for(size_t j = 0, valueCount = option.value.size(); j < valueCount; ++j)
      {
        settings->add_value(option.string_key, option.value[j]);
      }
    }
  }
}

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

#if WITH_LIBROYALE
  // Probe for a PicoFlexx camera.
  if(cameraSubengine == NULL)
  {
    std::cout << "[spaint] Probing PicoFlexx camera\n";
    cameraSubengine = check_camera_subengine(new PicoFlexxEngine(""));
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

#ifdef WITH_ZED
  // Probe for a Zed camera.
  if(cameraSubengine == NULL)
  {
    std::cout << "[spaint] Probing Zed camera\n";
    cameraSubengine = check_camera_subengine(new ZedImageSourceEngine(ZedCamera::instance()));
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

    for(size_t j = 0; j < chunkCount; ++j)
    {
      if(chunks[j] == "InfiniTAM")
      {
        result += "<tracker type='infinitam'/>";
      }
      else if(chunks[j] == "Disk")
      {
        const std::string poseFileMask = (args.sequenceDirs[i] / "posem%06i.txt").string();
        result += "<tracker type='infinitam'><params>type=file,mask=" + poseFileMask + "</params></tracker>";
      }
      else
      {
        result += "<tracker type='import'><params>builtin:" + chunks[j] + "</params></tracker>";
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
 * \brief Parses a configuration file and adds its registered options to the application's variables map
 *        and its unregistered options to the application's settings.
 *
 * \param filename  The name of the configuration file.
 * \param options   The registered options for the application.
 * \param vm        The variables map for the application.
 * \param settings  The settings for the application.
 */
void parse_configuration_file(const std::string& filename, const po::options_description& options, po::variables_map& vm, const Settings_Ptr& settings)
{
  // Parse the options in the configuration file.
  po::parsed_options parsedConfigFileOptions = po::parse_config_file<char>(filename.c_str(), options, true);

  // Add any registered options to the variables map.
  po::store(parsedConfigFileOptions, vm);

  // Add any unregistered options to the settings.
  add_unregistered_options_to_settings(parsedConfigFileOptions, settings);
}

/**
 * \brief Post-process the program's command-line arguments and add them to the application settings.
 *
 * \param args      The program's command-line arguments.
 * \param options   The registered options for the application.
 * \param vm        The variables map for the application.
 * \param settings  The settings for the application.
 * \return          true, if the program should continue after post-processing its arguments, or false otherwise.
 */
bool postprocess_arguments(CommandLineArguments& args, const po::options_description& options, po::variables_map& vm, const Settings_Ptr& settings)
{
  // If the user specifies both sequence and explicit depth / RGB image mask flags, print an error message.
  if(!args.sequenceSpecifiers.empty() && (!args.depthImageMasks.empty() || !args.rgbImageMasks.empty()))
  {
    std::cout << "Error: Either sequence flags or explicit depth / RGB image mask flags may be specified, but not both.\n";
    return false;
  }

  // If the user specified a model to load, determine the model directory and parse the model's configuration file (if present).
  if(args.modelSpecifier != "")
  {
    args.modelDir = bf::is_directory(args.modelSpecifier) ? args.modelSpecifier : find_subdir_from_executable("models") / args.modelSpecifier / Model::get_world_scene_id();

    const bf::path configPath = *args.modelDir / "settings.ini";
    if(bf::is_regular_file(configPath))
    {
      // Parse any additional options from the model's configuration file.
      parse_configuration_file(configPath.string(), options, vm, settings);
      po::notify(vm);
    }
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

  // Add the post-processed arguments to the application settings.
  args.add_to_settings(settings);

  return true;
}

/**
 * \brief Parses any command-line arguments passed in by the user and adds them to the application settings.
 *
 * \param argc      The command-line argument count.
 * \param argv      The raw command-line arguments.
 * \param args      The parsed command-line arguments.
 * \param settings  The application settings.
 * \return          true, if the program should continue after parsing the command-line arguments, or false otherwise.
 */
bool parse_command_line(int argc, char *argv[], CommandLineArguments& args, const Settings_Ptr& settings)
{
  // Specify the possible options.
  po::options_description genericOptions("Generic options");
  genericOptions.add_options()
    ("help", "produce help message")
    ("batch", po::bool_switch(&args.batch), "enable batch mode")
    ("calib,c", po::value<std::string>(&args.calibrationFilename)->default_value(""), "calibration filename")
    ("cameraAfterDisk", po::bool_switch(&args.cameraAfterDisk), "switch to the camera after a disk sequence")
    ("configFile,f", po::value<std::string>(), "additional parameters filename")
    ("detectFiducials", po::bool_switch(&args.detectFiducials), "enable fiducial detection")
    ("experimentTag", po::value<std::string>(&args.experimentTag)->default_value(Settings::NOT_SET), "experiment tag")
    ("leapFiducialID", po::value<std::string>(&args.leapFiducialID)->default_value(""), "the ID of the fiducial to use for the Leap Motion")
    ("mapSurfels", po::bool_switch(&args.mapSurfels), "enable surfel mapping")
    ("noRelocaliser", po::bool_switch(&args.noRelocaliser), "don't use the relocaliser")
    ("pipelineType", po::value<std::string>(&args.pipelineType)->default_value("semantic"), "pipeline type")
    ("relocaliserType", po::value<std::string>(&args.relocaliserType)->default_value("forest"), "relocaliser type (ferns|forest|none)")
    ("renderFiducials", po::bool_switch(&args.renderFiducials), "enable fiducial rendering")
    ("saveMeshOnExit", po::bool_switch(&args.saveMeshOnExit), "save a mesh of the scene on exiting the application")
    ("saveModelsOnExit", po::bool_switch(&args.saveModelsOnExit), "save a model of each voxel scene on exiting the application")
    ("subwindowConfigurationIndex", po::value<std::string>(&args.subwindowConfigurationIndex)->default_value("1"), "subwindow configuration index")
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
    ("modelSpecifier,m", po::value<std::string>(&args.modelSpecifier)->default_value(""), "model specifier")
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

  // Parse the command line.
  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, options), vm);

  // If a configuration file was specified:
  if(vm.count("configFile"))
  {
    // Parse additional options from the configuration file and add any registered options to the variables map.
    // These will be post-processed (if necessary) and added to the settings later. Unregistered options are
    // also allowed: we add these directly to the settings without post-processing.
    parse_configuration_file(vm["configFile"].as<std::string>(), options, vm, settings);
  }

  po::notify(vm);

  // Post-process any registered options and add them to the settings.
  if(!postprocess_arguments(args, options, vm, settings)) return false;

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
  // Construct the settings object for the application. This is used to store both the
  // settings for InfiniTAM and our own extended settings. Note that we do not use the
  // tracker configuration string in the InfiniTAM settings, and so we set it to NULL.
  Settings_Ptr settings(new Settings);
  settings->trackerConfig = NULL;

  // Parse the command-line arguments.
  CommandLineArguments args;
  if(!parse_command_line(argc, argv, args, settings))
  {
    return 0;
  }

  // Initialise SDL.
  if(SDL_Init(SDL_INIT_VIDEO | SDL_INIT_JOYSTICK) < 0)
  {
    quit("Error: Failed to initialise SDL.");
  }

  // Find all available joysticks and report the number found to the user.
  const int availableJoysticks = SDL_NumJoysticks();
  std::cout << "[spaint] Found " << availableJoysticks << " joysticks.\n";

  // Open all available joysticks.
  typedef boost::shared_ptr<SDL_Joystick> SDL_Joystick_Ptr;
  std::vector<SDL_Joystick_Ptr> joysticks;
  for(int i = 0; i < availableJoysticks; ++i)
  {
    SDL_Joystick *joystick = SDL_JoystickOpen(i);
    if(!joystick) throw std::runtime_error("Couldn't open joystick " + boost::lexical_cast<std::string>(i));

    std::cout << "[spaint] Opened joystick " << i << ": " << SDL_JoystickName(joystick) << '\n';
    joysticks.push_back(SDL_Joystick_Ptr(joystick, &SDL_JoystickClose));
  }

#if defined(WITH_ARRAYFIRE) && defined(WITH_CUDA)
  // Tell ArrayFire to run on the primary GPU.
  afcu::setNativeId(0);
#endif

#ifdef WITH_GLUT
  // Initialise GLUT (used for text rendering only).
  glutInit(&argc, argv);
#endif

#ifdef WITH_OVR
  // If we built with Rift support, initialise the Rift SDK.
  ovr_Initialize();
#endif

  if(args.cameraAfterDisk || !args.noRelocaliser) settings->behaviourOnFailure = ITMLibSettings::FAILUREMODE_RELOCALISE;

  // Pass the device type to the memory block factory.
  MemoryBlockFactory::instance().set_device_type(settings->deviceType);

  // Construct the image source engine.
  boost::shared_ptr<CompositeImageSourceEngine> imageSourceEngine(new CompositeImageSourceEngine);

  // If a model was specified without either a disk sequence or the camera following it, add an idle subengine to allow the model to still be viewed.
  if(args.modelDir && args.depthImageMasks.empty() && !args.cameraAfterDisk)
  {
    const std::string calibrationFilename = (*args.modelDir / "calib.txt").string();
    imageSourceEngine->addSubengine(new IdleImageSourceEngine(calibrationFilename.c_str()));
  }

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

  // If no model and no disk sequences were specified, or we want to switch to the camera once all the disk sequences finish, add a camera subengine.
  if((!args.modelDir && args.depthImageMasks.empty()) || args.cameraAfterDisk)
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
  if(args.pipelineType == "slam")
  {
    pipeline.reset(new SLAMPipeline(
      settings,
      Application::resources_dir().string(),
      imageSourceEngine,
      make_tracker_config(args),
      mappingMode,
      trackingMode,
      args.modelDir,
      fiducialDetector,
      args.detectFiducials
    ));
  }
  else if(args.pipelineType == "semantic")
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
      args.modelDir,
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
  else throw std::runtime_error("Unknown pipeline type: " + args.pipelineType);

#ifdef WITH_LEAP
  // Set the ID of the fiducial to use for the Leap Motion (if any).
  pipeline->get_model()->set_leap_fiducial_id(args.leapFiducialID);
#endif

  // Configure and run the application.
  Application app(pipeline, args.renderFiducials);
  app.set_batch_mode_enabled(args.batch);
  app.set_save_mesh_on_exit(args.saveMeshOnExit);
  app.set_save_models_on_exit(args.saveModelsOnExit);
  bool runSucceeded = app.run();

#ifdef WITH_OVR
  // If we built with Rift support, shut down the Rift SDK.
  ovr_Shutdown();
#endif

  // Close all open joysticks.
  joysticks.clear();

  // Shut down SDL.
  SDL_Quit();

  return runSucceeded ? EXIT_SUCCESS : EXIT_FAILURE;
}
catch(std::exception& e)
{
  std::cerr << e.what() << '\n';
  return EXIT_FAILURE;
}
