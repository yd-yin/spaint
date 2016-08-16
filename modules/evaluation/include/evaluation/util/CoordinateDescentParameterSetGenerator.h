/**
 * evaluation: CoordinateDescentParameterSetGenerator.h
 * Copyright (c) Torr Vision Group, University of Oxford, 2016. All rights reserved.
 */

#ifndef H_EVALUATION_COORDINATEDESCENTPARAMETERSETGENERATOR
#define H_EVALUATION_COORDINATEDESCENTPARAMETERSETGENERATOR

#include <string>
#include <utility>
#include <vector>

#include <boost/spirit/home/support/detail/hold_any.hpp>

#include <tvgutil/numbers/RandomNumberGenerator.h>

#include "../core/ParamSetUtil.h"

namespace evaluation {

/**
 * \brief An instance of this class will try to find the parameters that minimise a function using coordinate descent.
 */
class CoordinateDescentParameterSetGenerator
{
  //#################### PRIVATE VARIABLES ####################
private:
  /** The best parameter indices along each parameter dimension. */
  mutable std::vector<size_t> m_bestParamIndices;

  /** The best parameter indices over all epochs. */
  mutable std::vector<size_t> m_bestParamIndicesAllTime;

  /** The best score within the current epoch. */
  mutable float m_bestScore;

  /** The best score over all epochs. */
  mutable float m_bestScoreAllTime;

  /** The dimension index along which to search. */
  mutable size_t m_currentDimIndex;

  /** The current parameter indices for which a score is needed. */
  mutable std::vector<size_t> m_currentParamIndices;

  /** The set of parameters for which a score is needed. */
  mutable ParamSet m_currentParamSet;

  /** The total number of parameter dimensions. */
  size_t m_dimCount;

  /** The number of passes that should be made through the parameter set. */
  size_t m_epochCount;

  /** The first dimension along which to search. */
  mutable size_t m_firstDimIndex;

  /** A flag indicating whether the parameter generator has passed the first iteration. */
  mutable bool m_firstIteration;

  /** The total number of parameters that need to be searched in one epoch. */
  size_t m_globalParamCount;

  /** A list of the scores associated with each parameter. */
  mutable std::vector<std::vector<float> > m_paramScores;

  /** A list of the possible values for each parameter (e.g. [("A", [1,2]), ("B", [3,4])]). */
  std::vector<std::pair<std::string,std::vector<boost::spirit::hold_any> > > m_paramValues;

  /** A record of the best parameter indices obtained in the last epoch. */
  mutable std::vector<size_t> m_previousBestParamIndices;

  /** The random number generator to use when deciding which parameters to search first. */
  mutable tvgutil::RandomNumberGenerator m_rng;

  /** The total number of independent parameters that have only one value. */
  size_t m_singlePointParamCount;

  //#################### CONSTRUCTORS ####################
public:
  CoordinateDescentParameterSetGenerator(unsigned int seed, size_t epochCount);

  //#################### PUBLIC MEMBER FUNCTIONS ####################
public:
  /**
   * \brief Adds a parameter, together with a list of the values it may assume.
   *
   * \param param   The parameter name.
   * \param values  The values the parameter may assume.
   * \return        The generator itself (so that calls to add_param may be chained).
   */
  CoordinateDescentParameterSetGenerator& add_param(const std::string& param, const std::vector<boost::spirit::hold_any>& values);

  /**
   * \brief Gets the best parameter set over all epochs.
   *
   * \return  The best parameters set.
   */
  ParamSet get_best_param_set() const;

  /**
   * \brief Gets the best score over all epochs.
   *
   * \return  The best score.
   */
  float get_best_score() const;

  /**
   * \brief Gets the total number of iterations.
   *
   * \return The total number of iterations.
   */
  size_t get_iteration_count() const;

  /**
   * \brief Gets the next parameter set to evaluate.
   *
   * \return  The next parameter set.
   */
  ParamSet get_next_param_set() const;

  /**
   * \brief Initialise the coordinate descent parameter generator.
   */
  void initialise();

  /**
   * \brief Adds a score corresponding to a set of parameters.
   *
   * \param paramSet  The parameter set.
   * \param score     The score.
   */
  void score_param_set_and_update_state(const ParamSet& paramSet, float score) const;

  /**
   * \brief Output the parameters and the associated values they may assume.
   *
   * \param os  The stream.
   */
  void output_param_values(std::ostream& os) const;

  //#################### PRIVATE MEMBER FUNCTIONS ####################
private:
  /**
   * \brief Converts a set of parameter indices to a parameter set.
   *
   * \param paramIndices  The index set.
   * \return              The parameter set corresponding to the index set.
   */
  ParamSet param_indices_to_set(const std::vector<size_t>& paramIndices) const;

  /**
   * \brief Randomly restart the parameter search when learning has converged.
   */
  void random_restart() const;

  /**
   * \brief Update the current state of the parameter generator.
   */
  void update_state() const;
};

}

#endif
