/**
 * spaint: VOPFeatureCalculator.h
 */

#ifndef H_SPAINT_VOPFEATURECALCULATOR
#define H_SPAINT_VOPFEATURECALCULATOR

#include "FeatureCalculator.h"

namespace spaint {

/**
 * \brief An instance of this class can be used to calculate VOP feature descriptors for voxels sampled from a scene.
 */
class VOPFeatureCalculator : public FeatureCalculator
{
  //#################### PROTECTED VARIABLES ####################
protected:
  /** The maximum number of labels that can be in use. */
  size_t m_maxLabelCount;

  /** The maximum number of voxels that have been sampled for each label. */
  size_t m_maxVoxelsPerLabel;

  /** The side length of a VOP patch (must be odd). */
  size_t m_patchSize;

  /** The spacing in the scene between individual pixels in a patch. */
  float m_patchSpacing;

  /** The surface normals at the voxel locations. */
  mutable ORUtils::MemoryBlock<Vector3f> m_surfaceNormalsMB;

  /** The x axes of the coordinate systems in the tangent planes to the surfaces at the voxel locations. */
  mutable ORUtils::MemoryBlock<Vector3f> m_xAxesMB;

  /** The y axes of the coordinate systems in the tangent planes to the surfaces at the voxel locations. */
  mutable ORUtils::MemoryBlock<Vector3f> m_yAxesMB;

  //#################### CONSTRUCTORS ####################
protected:
  /**
   * \brief Constructs a VOP feature calculator.
   *
   * \param maxLabelCount     The maximum number of labels that can be in use.
   * \param maxVoxelsPerLabel The maximum number of voxels that have been sampled for each label.
   * \param patchSize         The side length of a VOP patch (must be odd).
   * \param patchSpacing      The spacing in the scene between individual pixels in a patch.
   */
  VOPFeatureCalculator(size_t maxLabelCount, size_t maxVoxelsPerLabel, size_t patchSize, float patchSpacing);

  //#################### PRIVATE ABSTRACT MEMBER FUNCTIONS ####################
private:
  /**
   * \brief Calculates the surface normals at the voxel locations.
   *
   * \param voxelLocationsMB        A memory block containing the locations of the voxels for which to calculate the surface normals (grouped by label).
   * \param voxelCountsForLabelsMB  A memory block containing the numbers of voxels for each label.
   * \param voxelData               The scene's voxel data.
   * \param indexData               The scene's index data.
   */
  virtual void calculate_surface_normals(const ORUtils::MemoryBlock<Vector3s>& voxelLocationsMB,
                                         const ORUtils::MemoryBlock<unsigned int>& voxelCountsForLabelsMB,
                                         const SpaintVoxel *voxelData, const ITMVoxelIndex::IndexData *indexData) const = 0;

  /**
   * \brief TODO
   */
  virtual void convert_patches_to_lab(ORUtils::MemoryBlock<float>& featuresMB) const = 0;

  /**
   * \brief Generates corodinate systems in the tangent planes to the surfaces at the voxel locations.
   *
   * \param voxelCountsForLabelsMB  A memory block containing the numbers of voxels for each label.
   */
  virtual void generate_coordinate_systems(const ORUtils::MemoryBlock<unsigned int>& voxelCountsForLabelsMB) const = 0;

  /**
   * \brief TODO
   */
  virtual void generate_rgb_patches(const ORUtils::MemoryBlock<Vector3s>& voxelLocationsMB,
                                    const ORUtils::MemoryBlock<unsigned int>& voxelCountsForLabelsMB,
                                    const SpaintVoxel *voxelData, const ITMVoxelIndex::IndexData *indexData,
                                    ORUtils::MemoryBlock<float>& featuresMB) const = 0;

  //#################### PUBLIC MEMBER FUNCTIONS ####################
public:
  /** Override */
  virtual void calculate_features(const ORUtils::MemoryBlock<Vector3s>& voxelLocationsMB,
                                  const ORUtils::MemoryBlock<unsigned int>& voxelCountsForLabelsMB,
                                  const ITMLib::Objects::ITMScene<SpaintVoxel,ITMVoxelIndex> *scene,
                                  ORUtils::MemoryBlock<float>& featuresMB) const;

  /** Override */
  virtual size_t get_feature_count() const;
};

}

#endif
