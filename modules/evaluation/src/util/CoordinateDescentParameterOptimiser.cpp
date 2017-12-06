/**
 * evaluation: CoordinateDescentParameterOptimiser.cpp
 * Copyright (c) Torr Vision Group, University of Oxford, 2016. All rights reserved.
 */

#include "util/CoordinateDescentParameterOptimiser.h"

#include <limits>

#include <boost/assign/list_of.hpp>
#include <boost/lexical_cast.hpp>
using boost::assign::list_of;
using boost::spirit::hold_any;

namespace evaluation {

//#################### CONSTRUCTORS ####################

CoordinateDescentParameterOptimiser::CoordinateDescentParameterOptimiser(const CostFunction& costFunction, size_t epochCount, unsigned int seed)
: m_costFunction(costFunction), m_epochCount(epochCount), m_rng(seed)
{}

//#################### PUBLIC MEMBER FUNCTIONS ####################

CoordinateDescentParameterOptimiser& CoordinateDescentParameterOptimiser::add_param(const std::string& param, const std::vector<hold_any>& values)
{
  m_paramValues.push_back(std::make_pair(param, values));
  return *this;
}

ParamSet CoordinateDescentParameterOptimiser::optimise_for_parameters(float *bestCost) const
{
  std::vector<size_t> bestValueIndicesAllTime;
  float bestCostAllTime = std::numeric_limits<float>::max();

  // For each epoch:
  for(size_t i = 0; i < m_epochCount; ++i)
  {
    // Randomly generate an initial set of parameter value indices.
    std::vector<size_t> initialValueIndices = generate_random_value_indices();

    // Optimise the initial set of parameter value indices using coordinate descent.
    std::vector<size_t> optimisedValueIndices;
    float optimisedCost;
    boost::tie(optimisedValueIndices, optimisedCost) = perform_coordinate_descent(initialValueIndices);

    // If the optimised cost is the best we've seen so far, update the best cost and best parameter value indices.
    if(optimisedCost < bestCostAllTime)
    {
      bestCostAllTime = optimisedCost;
      bestValueIndicesAllTime = optimisedValueIndices;
    }
  }

  // Return the best parameters found and (optionally) the corresponding cost.
  if(bestCost) *bestCost = bestCostAllTime;
  return make_param_set(bestValueIndicesAllTime);
}

//#################### PRIVATE MEMBER FUNCTIONS ####################

float CoordinateDescentParameterOptimiser::compute_cost(const std::vector<size_t>& valueIndices) const
{
  return m_costFunction(make_param_set(valueIndices));
}

std::vector<size_t> CoordinateDescentParameterOptimiser::generate_random_value_indices() const
{
  std::vector<size_t> valueIndices;
  for(size_t i = 0, paramCount = m_paramValues.size(); i < paramCount; ++i)
  {
    valueIndices.push_back(m_rng.generate_int_from_uniform(0, static_cast<int>(m_paramValues[i].second.size()) - 1));
  }
  return valueIndices;
}

ParamSet CoordinateDescentParameterOptimiser::make_param_set(const std::vector<size_t>& valueIndices) const
{
  ParamSet paramSet;
  for(size_t i = 0; i < m_paramValues.size(); ++i)
  {
    std::string param = m_paramValues[i].first;
    std::string value = boost::lexical_cast<std::string>(m_paramValues[i].second[valueIndices[i]]);
    paramSet.insert(std::make_pair(param, value));
  }
  return paramSet;
}

std::pair<std::vector<size_t>,float> CoordinateDescentParameterOptimiser::perform_coordinate_descent(const std::vector<size_t>& initialValueIndices) const
{
  // Invariant: currentCost = compute_cost(currentValueIndices)

  // Initialise the current and best parameter value indices and their associated costs.
  std::vector<size_t> bestValueIndices, currentValueIndices;
  float bestCost, currentCost;
  bestValueIndices = currentValueIndices = initialValueIndices;
  bestCost = currentCost = compute_cost(currentValueIndices);

  // Pick the first parameter to optimise. The parameters will be optimised one-by-one, starting from this parameter.
  size_t paramCount = m_paramValues.size();
  int startingParamIndex = m_rng.generate_int_from_uniform(0, static_cast<int>(paramCount) - 1);

  // For each parameter, starting from the one just chosen:
  for(size_t k = 0; k < paramCount; ++k)
  {
    size_t paramIndex = (startingParamIndex + k) % paramCount;

    // Check how many possible values the parameter can take. If there's only one possibility, the parameter can be skipped.
    size_t valueCount = m_paramValues[paramIndex].second.size();
    if(valueCount == 1) continue;

    // Record the parameter value for which we already have the corresponding cost so that we can avoid re-evaluating it.
    size_t originalValueIndex = currentValueIndices[paramIndex];

    // For each possible new value that the parameter can take:
    std::vector<size_t> newValueIndices = currentValueIndices;
    for(size_t valueIndex = 0; valueIndex < valueCount; ++valueIndex)
    {
      // If we already know that the cost for the new value is no better than the cost for the current value, skip it.
      if(valueIndex == originalValueIndex) continue;

      // Compute the cost for the new value. If it's better than the cost for the current value, update the current value.
      newValueIndices[paramIndex] = valueIndex;
      float newCost = compute_cost(newValueIndices);
      if(newCost < currentCost)
      {
        currentValueIndices[paramIndex] = valueIndex;
        currentCost = newCost;
      }
    }

    // If the current cost is now the best, update the best parameter value indices and the associated cost.
    if(currentCost < bestCost)
    {
      bestValueIndices = currentValueIndices;
      bestCost = currentCost;
    }
  }

  return std::make_pair(bestValueIndices, bestCost);
}

}
