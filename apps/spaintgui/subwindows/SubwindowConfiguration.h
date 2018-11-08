/**
 * spaintgui: SubwindowConfiguration.h
 * Copyright (c) Torr Vision Group, University of Oxford, 2015. All rights reserved.
 */

#ifndef H_SPAINTGUI_SUBWINDOWCONFIGURATION
#define H_SPAINTGUI_SUBWINDOWCONFIGURATION

#include "Subwindow.h"

/**
 * \brief An instance of this class can be used to represent a configuration of sub-windows into which
 *        different types of scene visualisation can be rendered.
 */
class SubwindowConfiguration
{
  //#################### PRIVATE VARIABLES ####################
private:
  /** The subwindows in the configuration. */
  std::vector<Subwindow> m_subwindows;

  //#################### PUBLIC STATIC MEMBER FUNCTIONS ####################
public:
  /**
   * Makes a default sub-window configuration with the specified number of sub-windows.
   *
   * \param subwindowCount  The number of sub-windows the configuration should have (must be in the set {1,2,3}).
   * \param imgSize         The size of image needed to store the scene visualisation for each sub-window.
   * \param pipelineType    The type of pipeline being used.
   * \param agentPrefix     The collaborative agent prefix to use (Local or Remote).
   * \return                The sub-window configuration, if the sub-window count was valid, or null otherwise.
   */
  static boost::shared_ptr<SubwindowConfiguration> make_default(size_t subwindowCount, const Vector2i& imgSize, const std::string& pipelineType,
                                                                const std::string& agentPrefix = "Local");

  //#################### PUBLIC MEMBER FUNCTIONS ####################
public:
  /**
   * \brief Adds a sub-window to the configuration.
   *
   * \param subwindow The subwindow to add.
   */
  void add_subwindow(const Subwindow& subwindow);

  /**
   * \brief Computes the fractional position of the specified point in the window within the sub-window containing it (if any).
   *
   * \param fractionalWindowPos The fractional position of the point in the window (with components in the range [0,1]).
   * \return                    If the point is within a sub-window, then a pair, the first component of which is the index
   *                            of the sub-window containing the point, and the second component of which is the fractional
   *                            position of the point in the sub-window. If the point is not within a sub-window, then nothing.
   */
  boost::optional<std::pair<size_t,Vector2f> > compute_fractional_subwindow_position(const Vector2f& fractionalWindowPos) const;

  /**
   * \brief Gets the i'th sub-window in the configuration.
   *
   * \return  The i'th sub-window in the configuration.
   */
  Subwindow& subwindow(size_t i);

  /**
   * \brief Gets the i'th sub-window in the configuration.
   *
   * \return  The i'th sub-window in the configuration.
   */
  const Subwindow& subwindow(size_t i) const;

  /**
   * \brief Gets the number of sub-windows in the configuration.
   *
   * \return  The number of sub-windows in the configuration.
   */
  size_t subwindow_count() const;

  //#################### PRIVATE MEMBER FUNCTIONS ####################
private:
  /**
   * \brief Determines the index of the sub-window (if any) containing the specified point in the window.
   *
   * \param fractionalWindowPos The fractional position of the point in the window (with components in the range [0,1]).
   * \return                    The sub-window index (if any) of the point, or nothing otherwise.
   */
  boost::optional<size_t> determine_subwindow_index(const Vector2f& fractionalWindowPos) const;
};

//#################### TYPEDEFS ####################

typedef boost::shared_ptr<SubwindowConfiguration> SubwindowConfiguration_Ptr;
typedef boost::shared_ptr<const SubwindowConfiguration> SubwindowConfiguration_CPtr;

#endif
