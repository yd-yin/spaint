/**
 * spaintgui: Renderer.h
 * Copyright (c) Torr Vision Group, University of Oxford, 2015. All rights reserved.
 */

#ifndef H_SPAINTGUI_RENDERER
#define H_SPAINTGUI_RENDERER

// Prevent SDL from trying to define M_PI.
#define HAVE_M_PI

#include <SDL.h>

#include <spaint/ogl/WrappedGL.h>

#ifdef WITH_GLUT
#include <spaint/ogl/WrappedGLUT.h>
#endif

#include <rigging/MoveableCamera.h>

#include "../core/Model.h"
#include "../subwindows/SubwindowConfiguration.h"

/**
 * \brief An instance of a class deriving from this one can be used to render the spaint scene to a given target.
 */
class Renderer
{
  //#################### TYPEDEFS ####################
protected:
  typedef boost::shared_ptr<void> SDL_GLContext_Ptr;
  typedef boost::shared_ptr<SDL_Window> SDL_Window_Ptr;

  //#################### PRIVATE VARIABLES ####################
private:
  /** The OpenGL context for the window. */
  SDL_GLContext_Ptr m_context;

  /** A flag indicating whether or not to use median filtering when rendering the scene raycast. */
  bool m_medianFilteringEnabled;

  /** The spaint model. */
  Model_CPtr m_model;

  /** The sub-window configuration to use for visualising the scene. */
  SubwindowConfiguration_Ptr m_subwindowConfiguration;

  /** A flag indicating whether or not to use supersampling when rendering the scene raycast. */
  bool m_supersamplingEnabled;

  /** The ID of a texture in which to temporarily store the scene raycast and touch image when rendering. */
  GLuint m_textureID;

  /** The window into which to render. */
  SDL_Window_Ptr m_window;

  /** The size of the window's viewport. */
  Vector2i m_windowViewportSize;

  //#################### CONSTRUCTORS ####################
protected:
  /**
   * \brief Constructs a renderer.
   *
   * \param model                   The spaint model.
   * \param subwindowConfiguration  The sub-window configuration to use for visualising the scene.
   * \param windowViewportSize      The size of the window's viewport.
   */
  Renderer(const Model_CPtr& model, const SubwindowConfiguration_Ptr& subwindowConfiguration, const Vector2i& windowViewportSize);

  //#################### DESTRUCTOR ####################
public:
  /**
   * \brief Destroys the renderer.
   */
  virtual ~Renderer();

  //#################### PUBLIC ABSTRACT MEMBER FUNCTIONS ####################
public:
  /**
   * \brief Gets the monocular render state for the camera associated with the specified sub-window.
   *
   * If the camera is a stereo one, this will return the render state corresponding to the left eye.
   *
   * \param subwindowIndex  The index of the sub-window.
   * \return                The monocular render state for the camera associated with the sub-window.
   */
  virtual VoxelRenderState_CPtr get_monocular_render_state(size_t subwindowIndex) const = 0;

  /**
   * \brief Gets whether or not the renderer is rendering the scene in mono.
   *
   * \return  true, if the renderer is rendering the scene in mono, or false otherwise.
   */
  virtual bool is_mono() const = 0;

  /**
   * \brief Renders both the reconstructed scene and the synthetic scene from one or more camera poses.
   *
   * \param fracWindowPos   The fractional position of the mouse within the window's viewport.
   * \param renderFiducials Whether or not to render the fiducials (if any) that have been detected in the 3D scene.
   */
  virtual void render(const Vector2f& fracWindowPos, bool renderFiducials) const = 0;

  //#################### PUBLIC MEMBER FUNCTIONS ####################
public:
  /**
   * \brief Captures a screenshot that can be saved to disk.
   *
   * \return  The screenshot.
   */
  ITMUChar4Image_CPtr capture_screenshot() const;

  /**
   * \brief Computes the fractional position of point (x,y) in the window.
   *
   * \param x The x coordinate of the point whose fractional position is to be computed.
   * \param y The y coordinate of the point whose fractional position is to be computed.
   * \return  The fractional position of the specified point in the window.
   */
  Vector2f compute_fractional_window_position(int x, int y) const;

  /**
   * \brief Gets whether or not to use median filtering when rendering the scene raycast.
   *
   * \return  A flag indicating whether or not to use median filtering when rendering the scene raycast.
   */
  bool get_median_filtering_enabled() const;

  /**
   * \brief Gets the renderer's sub-window configuration.
   *
   * \return  The renderer's sub-window configuration.
   */
  const SubwindowConfiguration_Ptr& get_subwindow_configuration();

  /**
   * \brief Gets the renderer's sub-window configuration.
   *
   * \return  The renderer's sub-window configuration.
   */
  SubwindowConfiguration_CPtr get_subwindow_configuration() const;

  /**
   * \brief Gets whether or not to use supersampling when rendering the scene raycast.
   *
   * \return  A flag indicating whether or not to use supersampling when rendering the scene raycast.
   */
  bool get_supersampling_enabled() const;

  /**
   * \brief Sets whether or not to use median filtering when rendering the scene raycast.
   *
   * \param medianFilteringEnabled  A flag indicating whether or not to use median filtering when rendering the scene raycast.
   */
  void set_median_filtering_enabled(bool medianFilteringEnabled);

  /**
   * \brief Sets whether or not to use supersampling when rendering the scene raycast.
   *
   * \param supersamplingEnabled  A flag indicating whether or not to use supersampling when rendering the scene raycast.
   */
  void set_supersampling_enabled(bool supersamplingEnabled);

  //#################### PROTECTED MEMBER FUNCTIONS ####################
protected:
  /**
   * \brief Sets appropriate projection and model-view matrices for 2D rendering.
   */
  static void begin_2d();

  /**
   * \brief Destroys the temporary image and texture used for visualising the scene.
   */
  void destroy_common();

  /**
   * \brief Restores the projection and model-view matrices that were active prior to 2D rendering.
   */
  static void end_2d();

  /**
   * \brief Gets the spaint model.
   *
   * \return  The spaint model.
   */
  Model_CPtr get_model() const;

  /**
   * \brief Gets the window into which to render.
   *
   * \return  The window into which to render.
   */
  SDL_Window *get_window() const;

  /**
   * \brief Gets the size of the window's viewport.
   *
   * \return  The size of the window's viewport.
   */
  const Vector2i& get_window_viewport_size() const;

  /**
   * \brief Initialises the temporary image and texture used for visualising the scene.
   */
  void initialise_common();

  /**
   * \brief Renders both the reconstructed scene and the synthetic scene.
   *
   * \param fracWindowPos       The fractional position of the mouse within the window's viewport.
   * \param renderFiducials     Whether or not to render the fiducials (if any) that have been detected in the 3D scene.
   * \param viewIndex           The index of the free camera view for the sub-window.
   * \param secondaryCameraName The name of the secondary camera from which to get the pose (if any).
   */
  void render_scene(const Vector2f& fracWindowPos, bool renderFiducials, int viewIndex = 0, const std::string& secondaryCameraName = "") const;

  /**
   * \brief Sets the window into which to render.
   *
   * \param window  The window into which to render.
   */
  void set_window(const SDL_Window_Ptr& window);

  //#################### PRIVATE MEMBER FUNCTIONS ####################
private:
  /**
   * \brief Generates a visualisation of the scene.
   *
   * \param output            The location into which to put the output image.
   * \param voxelScene        The voxel version of the scene to visualise.
   * \param surfelScene       The surfel version of the scene to visualise.
   * \param voxelRenderState  The voxel render state to use for intermediate storage (if relevant).
   * \param surfelRenderState The surfel render state to use for intermediate storage (if relevant).
   * \param pose              The pose from which to visualise the scene (if relevant).
   * \param view              The current view of the scene.
   * \param intrinsics        The intrinsics to use when rendering synthetic visualisations of the scene.
   * \param visualisationType The type of visualisation to generate.
   * \param surfelFlag        Whether or not to render a surfel visualisation rather than a voxel one.
   * \param postprocessor     An optional function with which to postprocess the visualisation before returning it.
   */
  void generate_visualisation(const ITMUChar4Image_Ptr& output, const spaint::SpaintVoxelScene_CPtr& voxelScene, const spaint::SpaintSurfelScene_CPtr& surfelScene,
                              VoxelRenderState_Ptr& voxelRenderState, SurfelRenderState_Ptr& surfelRenderState, const ORUtils::SE3Pose& pose, const View_CPtr& view,
                              const ITMLib::ITMIntrinsics& intrinsics, spaint::VisualisationGenerator::VisualisationType visualisationType, bool surfelFlag,
                              const boost::optional<spaint::VisualisationGenerator::Postprocessor>& postprocessor) const;

  /**
   * \brief Renders a semi-transparent colour overlay over the existing scene.
   *
   * \param overlay The colour overlay.
   */
  void render_overlay(const ITMUChar4Image_CPtr& overlay) const;

#if WITH_GLUT && USE_PIXEL_DEBUGGING
  /**
   * \brief Renders the value of a pixel in the specified sub-window.
   *
   * \param fracWindowPos The fractional position of the mouse within the window's viewport.
   * \param subwindow     The sub-window.
   */
  void render_pixel_value(const Vector2f& fracWindowPos, const Subwindow& subwindow) const;
#endif

  /**
   * \brief Renders the specified reconstructed scene into a sub-window.
   *
   * \param sceneID   The scene ID.
   * \param pose      The camera pose.
   * \param subwindow The sub-window into which to render.
   * \param viewIndex The index of the free camera view for the sub-window.
   */
  void render_reconstructed_scene(const std::string& sceneID, const ORUtils::SE3Pose& pose, Subwindow& subwindow, int viewIndex) const;

  /**
   * \brief Renders a synthetic scene to augment what actually exists in the real world.
   *
   * \param sceneID         The ID of the reconstructed scene to augment.
   * \param pose            The camera pose.
   * \param cameraMode      The camera mode.
   * \param renderFiducials Whether or not to render the fiducials (if any) that have been detected in the 3D scene.
   */
  void render_synthetic_scene(const std::string& sceneID, const ORUtils::SE3Pose& pose, Subwindow::CameraMode cameraMode, bool renderFiducials) const;

#ifdef WITH_GLUT
  /**
   * \brief Renders some text in a specified colour and at a specified position.
   *
   * \param text    The text to render.
   * \param colour  The colour in which to render the text.
   * \param pos     The position at which to render the text.
   * \param font    The font in which to render the text.
   */
  static void render_text(const std::string& text, const Vector3f& colour, const Vector2f& pos, void *font = GLUT_BITMAP_HELVETICA_18);
#endif

  /**
   * \brief Renders a quad textured with the specified texture.
   *
   * \param textureID The ID of the texture to apply to the quad.
   */
  static void render_textured_quad(GLuint textureID);

  /**
   * \brief Sets the OpenGL projection matrix based on a set of intrinsic camera parameters.
   *
   * \param intrinsics  The intrinsic camera parameters.
   * \param width       The width of the viewport.
   * \param height      The height of the viewport.
   */
  static void set_projection_matrix(const ITMLib::ITMIntrinsics& intrinsics, int width, int height);

  //#################### FRIENDS ####################

  friend class SelectorRenderer;
};

#endif
