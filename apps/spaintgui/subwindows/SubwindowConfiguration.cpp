/**
 * spaintgui: SubwindowConfiguration.cpp
 * Copyright (c) Torr Vision Group, University of Oxford, 2015. All rights reserved.
 */

#include "SubwindowConfiguration.h"

//#################### PUBLIC STATIC MEMBER FUNCTIONS ####################

SubwindowConfiguration_Ptr SubwindowConfiguration::make_default(size_t subwindowCount, const Vector2i& imgSize)
{
  SubwindowConfiguration_Ptr config;
  if(subwindowCount > 0) config.reset(new SubwindowConfiguration);

  switch(subwindowCount)
  {
    case 1:
    {
      config->add_subwindow(Subwindow(Vector2f(0, 0), Vector2f(1, 1), Raycaster::RT_SEMANTICLAMBERTIAN, imgSize));
      break;
    }
    case 2:
    {
      const float x = 0.5f;
      config->add_subwindow(Subwindow(Vector2f(0, 0), Vector2f(x, 1), Raycaster::RT_SEMANTICLAMBERTIAN, imgSize));
      config->add_subwindow(Subwindow(Vector2f(x, 0), Vector2f(1, 1), Raycaster::RT_SEMANTICCOLOUR, imgSize));
      break;
    }
    case 3:
    {
      const float x = 0.665f;
      const float y = 0.5f;
      config->add_subwindow(Subwindow(Vector2f(0, 0), Vector2f(x, y * 2), Raycaster::RT_SEMANTICLAMBERTIAN, imgSize));
      config->add_subwindow(Subwindow(Vector2f(x, 0), Vector2f(1, y), Raycaster::RT_SEMANTICCOLOUR, imgSize));
      config->add_subwindow(Subwindow(Vector2f(x, y), Vector2f(1, y * 2), Raycaster::RT_SEMANTICPHONG, imgSize));
      break;
    }
    default:
      break;
  }

  return config;
}

//#################### PUBLIC MEMBER FUNCTIONS ####################

void SubwindowConfiguration::add_subwindow(const Subwindow& subwindow)
{
  m_subwindows.push_back(subwindow);
}

boost::optional<Vector2f> SubwindowConfiguration::compute_fractional_subwindow_position(const Vector2f& fractionalViewportPos) const
{
  boost::optional<size_t> subwindowIndex = determine_subwindow_index(fractionalViewportPos);
  if(!subwindowIndex) return boost::none;

  const Subwindow& subwindow = m_subwindows[*subwindowIndex];
  const Vector2f& tl = subwindow.top_left();

  return Vector2f(
    CLAMP((fractionalViewportPos.x - tl.x) / subwindow.width(), 0.0f, 1.0f),
    CLAMP((fractionalViewportPos.y - tl.y) / subwindow.height(), 0.0f, 1.0f)
  );
}

boost::optional<size_t> SubwindowConfiguration::determine_subwindow_index(const Vector2f& fractionalViewportPos) const
{
  for(size_t i = 0, count = m_subwindows.size(); i < count; ++i)
  {
    const Subwindow& subwindow = m_subwindows[i];
    const Vector2f& tl = subwindow.top_left();
    const Vector2f& br = subwindow.bottom_right();
    if(tl.x <= fractionalViewportPos.x && fractionalViewportPos.x <= br.x &&
       tl.y <= fractionalViewportPos.y && fractionalViewportPos.y <= br.y)
    {
      return i;
    }
  }

  return boost::none;
}

Subwindow& SubwindowConfiguration::subwindow(size_t i)
{
  return m_subwindows[i];
}

const Subwindow& SubwindowConfiguration::subwindow(size_t i) const
{
  return m_subwindows[i];
}

size_t SubwindowConfiguration::subwindow_count() const
{
  return m_subwindows.size();
}
