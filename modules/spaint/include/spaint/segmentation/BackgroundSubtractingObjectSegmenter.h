/**
 * spaint: BackgroundSubtractingObjectSegmenter.h
 * Copyright (c) Torr Vision Group, University of Oxford, 2016. All rights reserved.
 */

#ifndef H_SPAINT_BACKGROUNDSUBTRACTINGOBJECTSEGMENTER
#define H_SPAINT_BACKGROUNDSUBTRACTINGOBJECTSEGMENTER

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "ColourAppearanceModel.h"
#include "Segmenter.h"
#include "../touch/TouchDetector.h"

namespace spaint {

/**
 * \brief An instance of this class can be used to segment an object that is placed in front of a static scene
 *        using background subtraction.
 */
class BackgroundSubtractingObjectSegmenter : public Segmenter
{
  //#################### PRIVATE VARIABLES ####################
private:
  /** The colour appearance model to use to separate the user's hand from any object it's holding. */
  ColourAppearanceModel_Ptr m_handAppearanceModel;

  /** The touch detector to use to make the change and hand masks. */
  mutable TouchDetector_Ptr m_touchDetector;

  //#################### CONSTRUCTORS ####################
public:
  /**
   * \brief Constructs a background-subtracting object segmenter.
   *
   * \param view          The current view of the scene.
   * \param itmSettings   The settings to use for InfiniTAM.
   * \param touchSettings The settings to use for the touch detector.
   */
  BackgroundSubtractingObjectSegmenter(const View_CPtr& view, const Settings_CPtr& itmSettings, const TouchSettings_Ptr& touchSettings);

  //#################### PUBLIC MEMBER FUNCTIONS ####################
public:
  /** Override */
  virtual void reset();

  /** Override */
  virtual ITMUCharImage_CPtr segment(const ORUtils::SE3Pose& pose, const RenderState_CPtr& renderState) const;

  /** Override */
  virtual ITMUChar4Image_CPtr train(const ORUtils::SE3Pose& pose, const RenderState_CPtr& renderState);

  //#################### PRIVATE MEMBER FUNCTIONS ####################
private:
  /**
   * \brief Makes a mask of any changes in the scene with respect to the reconstructed model.
   *
   * \param depthInput  The live depth input from the camera.
   * \param pose        The camera pose from which the scene is being viewed.
   * \param renderState The render state corresponding to the camera.
   */
  ITMUCharImage_CPtr make_change_mask(const ITMFloatImage_CPtr& depthInput, const ORUtils::SE3Pose& pose, const RenderState_CPtr& renderState) const;

  /**
   * \brief Makes a mask denoting the location of the user's hand as seen from the camera.
   *
   * \param depthInput  The live depth input from the camera.
   * \param pose        The camera pose from which the scene is being viewed.
   * \param renderState The render state corresponding to the camera.
   */
  ITMUCharImage_CPtr make_hand_mask(const ITMFloatImage_CPtr& depthInput, const ORUtils::SE3Pose& pose, const RenderState_CPtr& renderState) const;

  //#################### PRIVATE STATIC MEMBER FUNCTIONS ####################
private:
  /**
   * \brief Updates a mask to retain only connected components over a certain size.
   *
   * \param mask                  The mask to update.
   * \param minimumComponentSize  The minimum size of component to retain.
   */
  static void remove_small_components(cv::Mat1b& mask, int minimumComponentSize);
};

}

#endif
