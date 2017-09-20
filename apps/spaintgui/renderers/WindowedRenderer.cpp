/**
 * spaintgui: WindowedRenderer.cpp
 * Copyright (c) Torr Vision Group, University of Oxford, 2015. All rights reserved.
 */

#include "WindowedRenderer.h"
using namespace ORUtils;

#include <stdexcept>

//#################### CONSTRUCTORS ####################

WindowedRenderer::WindowedRenderer(const std::string& title, const Model_CPtr& model, const SubwindowConfiguration_Ptr& subwindowConfiguration,
                                   const Vector2i& windowViewportSize)
: Renderer(model, subwindowConfiguration, windowViewportSize)
{
  // Create the window into which to render.
  SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
  SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);

  set_window(SDL_Window_Ptr(
    SDL_CreateWindow(
      title.c_str(),
      SDL_WINDOWPOS_UNDEFINED,
      SDL_WINDOWPOS_UNDEFINED,
      windowViewportSize.width,
      windowViewportSize.height,
      SDL_WINDOW_OPENGL
    ),
    &SDL_DestroyWindow
  ));

  // Initialise the temporary image and texture used for visualising the scene.
  initialise_common();
}

//#################### DESTRUCTOR ####################

WindowedRenderer::~WindowedRenderer()
{
  destroy_common();
}

//#################### PUBLIC MEMBER FUNCTIONS ####################

VoxelRenderState_CPtr WindowedRenderer::get_monocular_render_state(size_t subwindowIndex) const
{
  return get_subwindow_configuration()->subwindow(subwindowIndex).get_voxel_render_state();
}

bool WindowedRenderer::is_mono() const
{
  return true;
}

void WindowedRenderer::render(const Vector2f& fracWindowPos, bool renderFiducials) const
{
#if USE_FOCUS_REACQUISITION
  // Reacquire the focus for this window if it has been lost to debugging windows.
  SDL_RaiseWindow(get_window());
#endif

  // Render the scene.
  render_scene(fracWindowPos, renderFiducials);

  // Swap the front and back buffers.
  SDL_GL_SwapWindow(get_window());
}
