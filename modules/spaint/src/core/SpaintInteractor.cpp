/**
 * spaint: SpaintInteractor.cpp
 */

#include "core/SpaintInteractor.h"

#include "markers/cpu/VoxelMarker_CPU.h"
#ifdef WITH_CUDA
#include "markers/cuda/VoxelMarker_CUDA.h"
#endif
#include "selectors/PickingSelector.h"

namespace spaint {

//#################### CONSTRUCTORS ####################

SpaintInteractor::SpaintInteractor(const SpaintModel_Ptr& model)
: m_model(model),
  m_selector(new PickingSelector(model->get_settings())),
  m_semanticLabel(1)
{
  // Set up the voxel marker.
  if(model->get_settings()->deviceType == ITMLibSettings::DEVICE_CUDA)
  {
#ifdef WITH_CUDA
    // Use the CUDA implementation.
    m_voxelMarker.reset(new VoxelMarker_CUDA);
#else
    // This should never happen as things stand - we set deviceType to DEVICE_CPU if CUDA support isn't available.
    throw std::runtime_error("Error: CUDA support not currently available. Reconfigure in CMake with the WITH_CUDA option set to on.");
#endif
  }
  else
  {
    // Use the CPU implementation.
    m_voxelMarker.reset(new VoxelMarker_CPU);
  }
}

//#################### PUBLIC MEMBER FUNCTIONS ####################

SpaintInteractor::Selection_CPtr SpaintInteractor::get_selection() const
{
  return m_selector ? m_selector->get_selection() : Selection_CPtr();
}

Selector_CPtr SpaintInteractor::get_selector() const
{
  return m_selector;
}

unsigned char SpaintInteractor::get_semantic_label() const
{
  return m_semanticLabel;
}

void SpaintInteractor::mark_voxels(const Selection_CPtr& selection, unsigned char label)
{
  m_voxelMarker->mark_voxels(*selection, label, m_model->get_scene().get());
}

bool SpaintInteractor::selector_is_active() const
{
  return m_selector && m_selector->is_active();
}

void SpaintInteractor::set_semantic_label(unsigned char semanticLabel)
{
  m_semanticLabel = semanticLabel;
}

void SpaintInteractor::update_selector(const InputState& inputState, const RenderState_CPtr& renderState)
{
  // Allow the user to switch between different selectors.
  if(inputState.key_down(SDLK_i))
  {
    if(inputState.key_down(SDLK_1)) m_selector.reset();
    else if(inputState.key_down(SDLK_2)) m_selector.reset(new PickingSelector(m_model->get_settings()));
  }

  // Update the current selector.
  if(m_selector) m_selector->update(inputState, renderState);
}

}
