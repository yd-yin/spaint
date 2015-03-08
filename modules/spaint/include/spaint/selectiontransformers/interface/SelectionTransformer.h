/**
 * spaint: SelectionTransformer.h
 */

#ifndef H_SPAINT_SELECTIONTRANSFORMER
#define H_SPAINT_SELECTIONTRANSFORMER

#include <ITMLib/Objects/ITMScene.h>
#include <ITMLib/Utils/ITMLibSettings.h>

namespace spaint {

/**
 * \brief An instance of a class deriving from this one can be used to transform one selection of voxels in the scene into another.
 */
class SelectionTransformer
{
  //#################### TYPEDEFS ####################
public:
  typedef ORUtils::MemoryBlock<Vector3s> Selection;

  //#################### DESTRUCTOR ####################
public:
  /**
   * \brief Destroys the selection transformer.
   */
  virtual ~SelectionTransformer();

  //#################### PUBLIC ABSTRACT MEMBER FUNCTIONS ####################
public:
  /**
   * \brief Computes the size of the output selection of voxels corresponding to the specified input selection.
   *
   * \param inputSelectionMB  A memory block containing the input selection of voxels.
   * \return                  The size of the output selection of voxels corresponding to the specified input selection.
   */
  virtual int compute_output_selection_size(const Selection& inputSelectionMB) const = 0;

  /**
   * \brief Transforms one selection of voxels in the scene into another.
   *
   * \param inputSelectionMB  A memory block containing the input selection of voxels.
   * \param outputSelectionMB A memory block into which to store the output selection of voxels.
   */
  virtual void transform_selection(const Selection& inputSelectionMB, Selection& outputSelectionMB) const = 0;

  //#################### PUBLIC MEMBER FUNCTIONS ####################
public:
  /**
   * \brief Transforms one selection of voxels in the scene into another.
   *
   * This function returns a raw pointer to memory allocated with new. It is the caller's responsibility
   * to ensure that this memory is eventually deleted. The easiest way to ensure this is to immediately
   * wrap the raw pointer in a shared_ptr when this function returns. This function can't use shared_ptr
   * because this header is included in a .cu file that's compiled by nvcc, and nvcc can't handle Boost.
   *
   * \param inputSelectionMB  A memory block containing the input selection of voxels.
   * \param deviceType        The device on which to perform the transformation.
   * \return                  A pointer to a memory block containing the output selection of voxels.
   */
  Selection *transform_selection(const Selection& inputSelectionMB, ITMLibSettings::DeviceType deviceType) const;
};

}

#endif
