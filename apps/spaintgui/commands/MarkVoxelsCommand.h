/**
 * spaintgui: MarkVoxelsCommand.h
 */

#ifndef H_SPAINTGUI_MARKVOXELSCOMMAND
#define H_SPAINTGUI_MARKVOXELSCOMMAND

#include <spaint/core/SpaintInteractor.h>

#include <tvgutil/commands/Command.h>

/**
 * \brief An instance of this class represents that can be used to mark voxels in the scene.
 */
class MarkVoxelsCommand : public tvgutil::Command
{
  //#################### PRIVATE VARIABLES ####################
private:
  /** The spaint interactor that is used to interact with the scene. */
  spaint::SpaintInteractor_Ptr m_interactor;

  /** The semantic label with which to mark the voxels. */
  unsigned char m_label;

  /** A memory block into which to store the old labels of the voxels being marked. */
  boost::shared_ptr<ORUtils::MemoryBlock<unsigned char> > m_oldVoxelLabelsMB;

  /** The locations of the voxels in the scene to mark. */
  boost::shared_ptr<const ORUtils::MemoryBlock<Vector3s> > m_voxelLocationsMB;

  //#################### CONSTRUCTORS ####################
public:
  /**
   * \brief Constructs a mark voxels command.
   *
   * \param voxelLocationsMB  The locations of the voxels in the scene to mark.
   * \param label             The semantic label with which to mark the voxels.
   * \param interactor        The spaint interactor that is used to interact with the scene.
   */
  MarkVoxelsCommand(const boost::shared_ptr<const ORUtils::MemoryBlock<Vector3s> >& voxelLocationsMB, unsigned char label,
                    const spaint::SpaintInteractor_Ptr& interactor);

  //#################### PUBLIC MEMBER FUNCTIONS ####################
public:
  /** Override */
  virtual void execute() const;

  /** Override */
  virtual void undo() const;
};

#endif
