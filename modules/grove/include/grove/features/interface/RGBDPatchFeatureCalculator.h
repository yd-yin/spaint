/**
 * grove: RGBDPatchFeatureCalculator.h
 * Copyright (c) Torr Vision Group, University of Oxford, 2017. All rights reserved.
 */

#ifndef H_GROVE_RGBDPATCHFEATURECALCULATOR
#define H_GROVE_RGBDPATCHFEATURECALCULATOR

#include <spaint/util/ITMImagePtrTypes.h>
#include <spaint/util/ITMMemoryBlockPtrTypes.h>

#include "../base/Descriptor.h"
#include "../../keypoints/Keypoint2D.h"
#include "../../keypoints/Keypoint3DColour.h"

namespace grove {

/**
 * \brief An instance of a class deriving from this one can be used to compute features based on depth and colour differences in RGBD images.
 *
 * The features are computed as described by Valentin et al. in "Exploiting Uncertainty in Regression Forests for Accurate Camera Relocalization".
 *
 * \param KeypointType    The type of keypoints computed by this class.
 * \param DescriptorType  The type of descriptors computed by this class.
 */
template <typename KeypointType, typename DescriptorType>
class RGBDPatchFeatureCalculator
{
  //#################### TYPEDEFS ####################
public:
  typedef ORUtils::Image<DescriptorType> DescriptorImage;
  typedef ORUtils::Image<KeypointType> KeypointImage;

  //#################### PROTECTED MEMBER VARIABLES ####################
protected:
  /** The number of depth features to compute from the depth image. */
  uint32_t m_depthFeatureCount;

  /** The offset in the descriptor after which we store the depth features. */
  uint32_t m_depthFeatureOffset;

  /** A memory block storing the offsets used to sample the depth values used in the descriptor. */
  ITMInt4MemoryBlock_Ptr m_depthOffsets;

  /** The step used to sample keypoints from the image. */
  uint32_t m_featureStep;

  /** Whether or not to normalize depth offsets by the depth associated to the corresponding keypoint. */
  bool m_normalizeDepth;

  /** Whether or not to normalize RGB offsets by the depth associated to the corresponding keypoint. */
  bool m_normalizeRgb;

  /** A memory block storing the colour channels associated to the RGB part of the descriptor. */
  ITMUCharMemoryBlock_Ptr m_rgbChannels;

  /** The number of colour features to compute from the RGB image. */
  uint32_t m_rgbFeatureCount;

  /** The offset in the descriptor after which we store the RGB features. */
  uint32_t m_rgbFeatureOffset;

  /** A memory block storing the offsets used to sample the colour pixels used in the descriptor. */
  ITMInt4MemoryBlock_Ptr m_rgbOffsets;

  //#################### CONSTRUCTORS ####################
protected:
  /**
   * \brief Constructs an RGBD patch feature calculator.
   *
   * Note: This is protected to force clients to make use of FeatureCalculatorFactory, which knows the correct values to use for the arguments.
   *
   * \param depthAdaptive      Whether or not to compute the depth-normalised version of the features.
   * \param depthFeatureCount  The number of features to compute from the depth image.
   * \param depthFeatureOffset The offset in the descriptor after which we store the depth features.
   * \param rgbFeatureCount    The number of features to compute from the RGB image.
   * \param rgbFeatureOffset   The offset in the descriptor after which we store the colour features.
   *
   * \throws std::invalid_argument If depthFeatureCount + rgbFeatureCount > DescriptorType::FEATURE_COUNT, or if the offsets cause out-of-bounds access.
   */
  RGBDPatchFeatureCalculator(bool depthAdaptive, uint32_t depthFeatureCount, uint32_t depthFeatureOffset, uint32_t rgbFeatureCount, uint32_t rgbFeatureOffset);

  //#################### DESTRUCTOR ####################
public:
  /**
   * \brief Destroys the RGBD patch feature calculator.
   */
  virtual ~RGBDPatchFeatureCalculator();

  //#################### PUBLIC ABSTRACT MEMBER FUNCTIONS ####################
public:
  /**
   * \brief Extract keypoints from an RGBD image and computes feature descriptors for them.
   *        Each keypoint position is in world coordinates, obtained by applying the cameraPose
   *        transformation to the 3D coordinates of the point in camera coordinates.
   *
   * \note  We implicitly assume that pixels in the colour and depth images are registered.
   * \note  Keypoints/descriptors are computed on a grid in the input image pair, with the step
   *        set by set_feature_step.
   *
   * \param rgbImage        The colour image.
   * \param depthImage      The depth image.
   * \param cameraPose      A transformation from the camera's reference frame to the
   *                        world reference frame (this will be applied to the 3D
   *                        keypoint positions in camera coordinates).
   * \param intrinsics      The intrinsic parameters of the depth camera.
   * \param keypointsImage  The output image that will contain the 3D coordinates
   *                        (in world reference frame, see cameraPose) and colour
   *                        of the extracted keypoints. Will be resized as necessary.
   * \param featuresImage   The output image that will contain the feature descriptors.
   */
  virtual void compute_feature(const ITMUChar4Image *rgbImage, const ITMFloatImage *depthImage,
                               const Matrix4f& cameraPose, const Vector4f& intrinsics,
                               KeypointImage *keypointsImage, DescriptorImage *featuresImage) const = 0;

  //#################### PUBLIC MEMBER FUNCTIONS ####################
public:
  /**
   * \brief Extract keypoints and compute the feature for an RGBD image.
   *        Each keypoint position is in camera coordinates, according to
   *        the depth camera intrinsics.
   *
   * \note  The implicit assumption is that pixels in the colour image and in
   *        the depth image are registered.
   * \note  Keypoint/Descriptors are computed on a grid in the input image pair,
   *        according to the step set by set_feature_step.
   *
   * \param rgbImage        The colour image.
   * \param depthImage      The depth image.
   * \param intrinsics      The intrinsic parameters of the depth camera.
   * \param keypointsImage  The output image that will contain the 3D coordinates
   *                        (in camera reference frame) and colour
   *                        of the extracted features. Will be resized as necessary.
   * \param featuresImage   The output image that will contain the descriptors.
   */
  void compute_feature(const ITMUChar4Image *rgbImage, const ITMFloatImage *depthImage, const Vector4f& intrinsics,
                       KeypointImage *keypointsImage, DescriptorImage *featuresImage) const;

  /**
   * \brief Gets the step used when selecting keypoints and computing the features.
   *
   * \return  The step used when selecting keypoints and computing the features.
   */
  uint32_t get_feature_step() const;

  /**
   * \brief Sets the step used when selecting keypoints and computing the features.
   *
   * \param featureStep The step used when selecting keypoints and computing the features.
   */
  void set_feature_step(uint32_t featureStep);

  //#################### PROTECTED MEMBER FUNCTIONS ####################
protected:
  /**
   * \brief Function used to validate whether the input images passed to compute_feature allow
   *        computation of the required features.
   *
   * \param rgbImage        The colour image.
   * \param depthImage      The depth image.
   *
   * \throws std::invalid_argument if the features cannot be computed.
   */
  void validate_input_images(const ITMUChar4Image *rgbImage, const ITMFloatImage *depthImage) const;
};

//#################### TYPEDEFS ####################

typedef Descriptor<256> RGBDPatchDescriptor;

typedef ORUtils::Image<RGBDPatchDescriptor> RGBDPatchDescriptorImage;
typedef boost::shared_ptr<RGBDPatchDescriptorImage> RGBDPatchDescriptorImage_Ptr;
typedef boost::shared_ptr<const RGBDPatchDescriptorImage> RGBDPatchDescriptorImage_CPtr;

typedef RGBDPatchFeatureCalculator<Keypoint2D, RGBDPatchDescriptor> RGBPatchFeatureCalculator;
typedef boost::shared_ptr<RGBPatchFeatureCalculator> RGBPatchFeatureCalculator_Ptr;
typedef boost::shared_ptr<const RGBPatchFeatureCalculator> RGBPatchFeatureCalculator_CPtr;

typedef RGBDPatchFeatureCalculator<Keypoint3DColour, RGBDPatchDescriptor> DA_RGBDPatchFeatureCalculator;
typedef boost::shared_ptr<DA_RGBDPatchFeatureCalculator> DA_RGBDPatchFeatureCalculator_Ptr;
typedef boost::shared_ptr<const DA_RGBDPatchFeatureCalculator> DA_RGBDPatchFeatureCalculator_CPtr;

}

#endif
