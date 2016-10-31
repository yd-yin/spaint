/**
 * spaintgui: Application.h
 * Copyright (c) Torr Vision Group, University of Oxford, 2015. All rights reserved.
 */

#ifndef H_SPAINTGUI_APPLICATION
#define H_SPAINTGUI_APPLICATION

#ifdef _MSC_VER
  // Suppress some VC++ warnings that are produced by boost/asio.hpp.
  #pragma warning(disable:4267 4996)
#endif

#include <boost/asio.hpp>

#ifdef _MSC_VER
  // Re-enable the VC++ warnings for the rest of the code.
  #pragma warning(default:4267 4996)
#endif

// Prevent SDL from trying to define M_PI.
#define HAVE_M_PI

#include <SDL.h>

#include <tvginput/InputState.h>

#include <tvgutil/commands/CommandManager.h>
#include <tvgutil/filesystem/SequentialPathGenerator.h>

#include "core/MultiScenePipeline.h"
#include "renderers/Renderer.h"

#ifdef WITH_OVR
#include "renderers/RiftRenderer.h"
#endif

/**
 * \brief The main application class for spaintgui.
 */
class Application
{
  //#################### TYPEDEFS ####################
private:
  typedef boost::shared_ptr<Renderer> Renderer_Ptr;

  //#################### PRIVATE VARIABLES ####################
private:
  /** The index of the sub-window with which the user is interacting. */
  size_t m_activeSubwindowIndex;

  /** Whether or not to terminate the application as soon as the last frame has been processeed. */
  bool m_runInBatch;

  /** The command manager. */
  tvgutil::CommandManager m_commandManager;

  /** The fractional position of the mouse within the window's viewport. */
  Vector2f m_fracWindowPos;

  /** The current state of the keyboard and mouse. */
  tvginput::InputState m_inputState;

  /** The mesh used by the meshing engine. */
  Mesh_Ptr m_mesh;

  /** The meshing engine. */
  MeshingEngine_Ptr m_meshingEngine;

  /** Whether or not to pause between frames (for debugging purposes). */
  bool m_pauseBetweenFrames;

  /** Whether or not the application is currently paused. */
  bool m_paused;

  /** The multi-scene pipeline that the application should use. */
  MultiScenePipeline_Ptr m_pipeline;

  /** The current renderer. */
  Renderer_Ptr m_renderer;

  /** Whether to save the reconstructed mesh on exit. */
  bool m_saveMeshOnExit;

  /** The path generator for the current sequence recording (if any). */
  boost::optional<tvgutil::SequentialPathGenerator> m_sequencePathGenerator;

  /** A set of sub-window configurations that the user can switch between as desired. */
  mutable std::vector<SubwindowConfiguration_Ptr> m_subwindowConfigurations;

  /** Whether or not to mirror poses between sub-windows that show the same scene. */
  bool m_usePoseMirroring;

  /** The path generator for the current video recording (if any). */
  boost::optional<tvgutil::SequentialPathGenerator> m_videoPathGenerator;

  /** The stream of commands being sent from the voice command server. */
  boost::asio::ip::tcp::iostream m_voiceCommandStream;

  //#################### CONSTRUCTORS ####################
public:
  /**
   * \brief Constructs the application.
   *
   * \param pipeline    The multi-scene pipeline that the application should use.
   * \param runInBatch  Whether the application should start running immediately
   *                    and terminate as soon as the last frame is processed.
   */
  explicit Application(const MultiScenePipeline_Ptr& pipeline, bool runInBatch);

  //#################### PUBLIC MEMBER FUNCTIONS ####################
public:
  /**
   * \brief Runs the application.
   */
  void run();

  /**
   * \brief Set whether to save the reconstructed mesh on exit.
   *
   * \param saveMesh True if the reconstructed mesh should be saved on exit.
   */
  void set_save_mesh_on_exit(bool saveMesh);

  //#################### PUBLIC STATIC MEMBER FUNCTIONS ####################
public:
  /**
   * \brief Gets the path to the resources directory.
   *
   * \return  The path to the resources directory.
   */
  static boost::filesystem::path resources_dir();

  //#################### PRIVATE MEMBER FUNCTIONS ####################
private:
  /**
   * \brief Gets the scene ID for the active sub-window.
   *
   * \return  The scene ID for the active sub-window.
   */
  const std::string& get_active_scene_id() const;

  /**
   * \brief Gets the sub-window with which the user is interacting.
   *
   * \return  The sub-window with which the user is interacting.
   */
  Subwindow& get_active_subwindow();

  /**
   * \brief Gets the sub-window with which the user is interacting.
   *
   * \return  The sub-window with which the user is interacting.
   */
  const Subwindow& get_active_subwindow() const;

  /**
   * \brief Gets the current monocular render state.
   *
   * If we're rendering in stereo, this will return the render state corresponding to the left eye.
   *
   * \return  The current monocular render state.
   */
  VoxelRenderState_CPtr get_monocular_render_state() const;

  /**
   * \brief Gets the specified sub-window configuration.
   *
   * \param i The index of the sub-window configuration to get.
   * \return  The specified sub-window configuration, if valid, or null otherwise.
   */
  SubwindowConfiguration_Ptr get_subwindow_configuration(size_t i) const;

  /**
   * \brief Handle key down events.
   *
   * \param keysym  A representation of the key that has been pressed.
   */
  void handle_key_down(const SDL_Keysym& keysym);

  /**
   * \brief Handle key up events.
   *
   * \param keysym  A representation of the key that has been released.
   */
  void handle_key_up(const SDL_Keysym& keysym);

  /**
   * \brief Handle mouse button down events.
   *
   * \param e The mouse button down event.
   */
  void handle_mousebutton_down(const SDL_MouseButtonEvent& e);

  /**
   * \brief Handle mouse button up events.
   *
   * \param e The mouse button up event.
   */
  void handle_mousebutton_up(const SDL_MouseButtonEvent& e);

  /**
   * \brief Processes user input that deals with the camera.
   */
  void process_camera_input();

  /**
   * \brief Processes user input that deals with commands (i.e. undo/redo).
   */
  void process_command_input();

  /**
   * \brief Processes any SDL events (e.g. those generated by user input).
   *
   * \return true, if the application should continue running, or false otherwise.
   */
  bool process_events();

  /**
   * \brief Takes action as relevant based on the current input state.
   */
  void process_input();

  /**
   * \brief Processes user input that deals with labelling the scene.
   */
  void process_labelling_input();

  /**
   * \brief Processes user input that deals with switching pipeline mode.
   */
  void process_mode_input();

  /**
   * \brief Processes user input that deals with switching the renderer or raycast type.
   */
  void process_renderer_input();

  /**
   * \brief Processes voice input from the user.
   */
  void process_voice_input();

  /**
   * \brief Saves the reconstruction to disk as a mesh.
   */
  void save_mesh() const;

  /**
   * \brief Saves a screenshot to disk.
   */
  void save_screenshot() const;

  /**
   * \brief Saves the next frame of the sequence being recorded to disk.
   */
  void save_sequence_frame();

  /**
   * \brief Saves the next frame of the video being recorded to disk.
   */
  void save_video_frame();

  /**
   * \brief Sets up the semantic labels with which the user can label the scene.
   */
  void setup_labels();

  /**
   * \brief Sets up the meshing engine if required.
   */
  void setup_meshing();

#ifdef WITH_OVR
  /**
   * \brief Switches to a Rift renderer.
   *
   * \param mode  The Rift rendering mode to use.
   */
  void switch_to_rift_renderer(RiftRenderer::RiftRenderingMode mode);
#endif

  /**
   * \brief Switches to a windowed renderer that uses the specified sub-window configuration.
   *
   * \param subwindowConfigurationIndex The index of the sub-window configuration to use.
   */
  void switch_to_windowed_renderer(size_t subwindowConfigurationIndex);

  /**
   * \brief Toggles sequence or video recording on or off.
   *
   * \param type          The type or recording (sequence or video).
   * \param pathGenerator The path generator associated with that type of recording.
   */
  void toggle_recording(const std::string& type, boost::optional<tvgutil::SequentialPathGenerator>& pathGenerator);
};

#endif
