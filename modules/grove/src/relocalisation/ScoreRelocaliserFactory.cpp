/**
 * grove: ScoreRelocaliserFactory.cpp
 * Copyright (c) Torr Vision Group, University of Oxford, 2017. All rights reserved.
 */

#include "relocalisation/ScoreRelocaliserFactory.h"

#include "relocalisation/cpu/ScoreRelocaliser_CPU.h"
#ifdef WITH_CUDA
#include "relocalisation/cuda/ScoreRelocaliser_CUDA.h"
#endif

using namespace tvgutil;

namespace grove {

//#################### PUBLIC STATIC MEMBER FUNCTIONS ####################

ScoreRelocaliser_Ptr ScoreRelocaliserFactory::make_score_relocaliser(DeviceType deviceType, const SettingsContainer_CPtr& settings, const std::string& forestFilename)
{
  ScoreRelocaliser_Ptr relocaliser;

  if(deviceType == DEVICE_CUDA)
  {
#ifdef WITH_CUDA
    relocaliser.reset(new ScoreRelocaliser_CUDA(forestFilename, settings));
#else
    throw std::runtime_error("Error: CUDA support not currently available. Reconfigure in CMake with the WITH_CUDA option set to on.");
#endif
  }
  else
  {
    relocaliser.reset(new ScoreRelocaliser_CPU(forestFilename, settings));
  }

  return relocaliser;
}

}
