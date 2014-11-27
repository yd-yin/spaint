/**
 * rafl: FeatureThresholdingDecisionFunctionGenerator.h
 */

#ifndef H_RAFL_FEATURETHRESHOLDINGDECISIONFUNCTIONGENERATOR
#define H_RAFL_FEATURETHRESHOLDINGDECISIONFUNCTIONGENERATOR

#include <cassert>

#include <tvgutil/RandomNumberGenerator.h>

#include "DecisionFunctionGenerator.h"
#include "FeatureThresholdingDecisionFunction.h"

namespace rafl {

/**
 * \brief An instance of an instantiation of this class template can be used to pick an appropriate decision function to split a set of examples.
 */
template <typename Label>
class FeatureThresholdingDecisionFunctionGenerator : public DecisionFunctionGenerator<Label>
{
  //#################### USINGS ####################
protected:
  using typename DecisionFunctionGenerator<Label>::Example_CPtr;

  //#################### PRIVATE VARIABLES ####################
private:
  /** A random number generator. */
  tvgutil::RandomNumberGenerator_Ptr m_randomNumberGenerator;

  //#################### CONSTRUCTORS ####################
public:
  /**
   * \brief Constructs a decision function generator that can randomly generate feature thresholding decision functions.
   *
   * \param randomNumberGenerator A random number generator.
   */
  explicit FeatureThresholdingDecisionFunctionGenerator(const tvgutil::RandomNumberGenerator_Ptr& randomNumberGenerator)
  : m_randomNumberGenerator(randomNumberGenerator)
  {}

  //#################### PRIVATE MEMBER FUNCTIONS ####################
private:
  /** Override */
  virtual DecisionFunction_Ptr generate_candidate_decision_function(const std::vector<Example_CPtr>& examples) const
  {
    assert(!examples.empty());

    int descriptorSize = static_cast<int>(examples[0]->get_descriptor()->size());

    // Pick a random feature in the descriptor to threshold.
    int featureIndex = m_randomNumberGenerator->generate_int_from_uniform(0, descriptorSize - 1);

    // Select an appropriate threshold by picking a random example and using
    // the value of the chosen feature from that example as the threshold.
    int exampleIndex = m_randomNumberGenerator->generate_int_from_uniform(0, static_cast<int>(examples.size()) - 1);
    float threshold = (*examples[exampleIndex]->get_descriptor())[featureIndex];

    return DecisionFunction_Ptr(new FeatureThresholdingDecisionFunction(featureIndex, threshold));
  }
};

}

#endif
