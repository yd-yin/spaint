/**
 * spaint: TouchSettings.cpp
 */

#include "touch/TouchSettings.h"

#include <boost/filesystem.hpp>

#include <tvgutil/DirectoryUtil.h>
#include <tvgutil/MapUtil.h>
#include <tvgutil/PropertyUtil.h>
#include <tvgutil/SerializationUtil.h>
using namespace tvgutil;

namespace spaint {

//#################### CONSTRUCTORS ####################

TouchSettings::TouchSettings(const boost::filesystem::path& touchSettingsFile)
: m_touchSettingsFile(touchSettingsFile)
{
  boost::property_tree::ptree tree = PropertyUtil::load_properties_from_xml(touchSettingsFile.string());
  initialise(PropertyUtil::make_property_map(tree));
}

//#################### PRIVATE MEMBER FUNCTIONS ####################

void TouchSettings::initialise(const std::map<std::string,std::string>& properties)
{
  std::string forestPath;

  #define GET_SETTING(param) tvgutil::MapUtil::typed_lookup(properties, #param, param);
    GET_SETTING(forestPath);
    GET_SETTING(lowerDepthThresholdMm);
    GET_SETTING(minCandidateFraction);
    GET_SETTING(minTouchAreaFraction);
    GET_SETTING(maxCandidateFraction);
    GET_SETTING(morphKernelSize);
    GET_SETTING(saveCandidateComponents);
    GET_SETTING(saveCandidateComponentsPath);
  #undef GET_SETTING

  boost::filesystem::path fullForestPath = m_touchSettingsFile.branch_path() / forestPath;
  if(!boost::filesystem::exists(fullForestPath)) throw std::runtime_error("Touch detection random forest not found: " + forestPath);

  forest = SerializationUtil::load_text(fullForestPath.string(), forest);

  if(saveCandidateComponents)
  {
    if(!boost::filesystem::is_directory(saveCandidateComponentsPath))
    {
      throw std::runtime_error("Save candidate components path not found: " + saveCandidateComponentsPath + ". Please add a valid path to the touch settings.");
    }

    static size_t fileCount = tvgutil::DirectoryUtil::get_file_count(saveCandidateComponentsPath);
    if(fileCount) throw std::runtime_error("Will not overwrite the " + boost::lexical_cast<std::string>(fileCount) + " images captured data in: " + saveCandidateComponentsPath);
  }
}

}
