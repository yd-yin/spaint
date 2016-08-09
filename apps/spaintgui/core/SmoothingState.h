/**
 * spaintgui: SmoothingState.h
 * Copyright (c) Torr Vision Group, University of Oxford, 2016. All rights reserved.
 */

#ifndef H_SPAINTGUI_SMOOTHINGSTATE
#define H_SPAINTGUI_SMOOTHINGSTATE

#include <spaint/smoothing/interface/LabelSmoother.h>

#include "Model.h"

/**
 * \brief TODO
 */
class SmoothingState
{
  //#################### DESTRUCTOR ####################
public:
  virtual ~SmoothingState() {}

  //#################### PUBLIC ABSTRACT MEMBER FUNCTIONS ####################
public:
  /**
   * \brief TODO
   */
  virtual const spaint::LabelSmoother_CPtr& get_label_smoother() const = 0;

  /**
   * \brief TODO
   */
  virtual const Model_Ptr& get_model() const = 0;
};

#endif
