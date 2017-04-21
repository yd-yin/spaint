/**
 * tvgutil: GlobalParameters.h
 * Copyright (c) Torr Vision Group, University of Oxford, 2017. All rights reserved.
 */

#ifndef H_TVGUTIL_GLOBALPARAMETERS
#define H_TVGUTIL_GLOBALPARAMETERS

#include <ostream>
#include <vector>

#include "../containers/MapUtil.h"

namespace tvgutil {

/**
 * \brief The singleton instance of this class can be used to access the global parameters used to configure an application.
 *
 * The parameters are represented as a key -> [value] map, i.e. there can be multiple values for the same named parameter.
 */
class GlobalParameters
{
  //#################### PRIVATE VARIABLES ####################
private:
  /** The key -> [value] map storing the values for the parameters. */
  std::map<std::string, std::vector<std::string> > m_params;

  //#################### SINGLETON IMPLEMENTATION ####################
private:
  /**
   * \brief Constructs the singleton instance.
   */
  GlobalParameters();

public:
  /**
   * \brief Gets the singleton instance.
   *
   * \return The singleton instance.
   */
  static GlobalParameters& instance();

  //#################### PUBLIC MEMBER FUNCTIONS ####################
public:
  /**
   * \brief Adds a value to the list of values for the specified parameter.
   *
   * \param key   The name of the parameter with which to associate the value.
   * \param value The value to add.
   */
  void add_value(const std::string& key, const std::string& value);

  /**
   * \brief Returns a typed value from the container.
   *
   * If multiple values have been added to the same key, the first is returned.
   *
   * \param key The key.
   * \return    The value.
   *
   * \throws std::runtime_error       If the container does not contain the specified key.
   * \throws boost::bad_lexical_cast  If the corresponding value in the container cannot be converted to the requested type.
   */
  template<typename T>
  T get_first_value(const std::string& key) const
  {
    const std::vector<std::string>& values = MapUtil::lookup(m_params, key);

    if(values.empty())
      throw std::runtime_error("Value for " + key + " not found in the container.");

    return boost::lexical_cast<T>(values[0]);
  }

  /**
   * \brief Returns a typed value from the container.
   *
   * If multiple values have been added to the same key, the first is returned.
   * If the key is missing returns the default value.
   *
   * \param key          The key.
   * \param defaultValue The default value.
   * \return             The value.
   *
   * \throws boost::bad_lexical_cast  If the corresponding value in the container cannot be converted to the requested type.
   */
  template<typename T>
  T get_first_value(const std::string& key, typename boost::mpl::identity<const T>::type& defaultValue) const
  {
    static std::vector<std::string> defaultEmptyVector;
    const std::vector<std::string>& values = MapUtil::lookup(m_params, key, defaultEmptyVector);

    return values.empty() ? defaultValue : boost::lexical_cast<T>(values[0]);
  }

  //#################### STREAM OPERATORS ####################
public:
  /**
   * \brief Outputs the global parameters to a stream.
   *
   * \param os  The stream to which to output the global parameters.
   * \param rhs The global parameters to output.
   * \return    The stream.
   */
  friend std::ostream& operator<<(std::ostream& os, const GlobalParameters& rhs);
};

//#################### TEMPLATE SPECIALIZATIONS ####################

/**
 * \brief Returns a typed value from the container.
 *
 * Specialization for bool since lexical_cast does not handle "true" and "false".
 * If multiple values have been added to the same key, the first is returned.
 *
 * \param key The key.
 * \return    The value.
 *
 * \throws std::runtime_error       If the container does not contain the specified key.
 * \throws boost::bad_lexical_cast  If the corresponding value in the container cannot be converted to the requested type.
 */
template<>
inline bool GlobalParameters::get_first_value<bool>(const std::string &key) const
{
  std::vector<std::string> values = MapUtil::lookup(m_params, key);

  if(values.empty())
    throw std::runtime_error("Value for " + key + " not found in the container.");

  bool value;
  std::istringstream ss(values[0]);
  ss >> std::boolalpha >> value;

  return value;
}

/**
 * \brief Returns a typed value from the container.
 *
 * Specialization for bool since lexical_cast does not handle "true" and "false".
 * If multiple values have been added to the same key, the first is returned.
 * If the key is missing returns the default value.
 *
 * \param key          The key.
 * \param defaultValue The default value.
 * \return             The value.
 *
 * \throws boost::bad_lexical_cast  If the corresponding value in the container cannot be converted to the requested type.
 */
template<>
inline bool GlobalParameters::get_first_value<bool>(const std::string &key, typename boost::mpl::identity<const bool>::type &defaultValue) const
{
  static std::vector<std::string> defaultEmptyVector;
  std::vector<std::string> values = MapUtil::lookup(m_params, key, defaultEmptyVector);

  bool value = defaultValue;

  if(!values.empty())
  {
    std::istringstream ss(values[0]);
    ss >> std::boolalpha >> value;
  }

  return value;
}

}

#endif
