/**
 * rafl: DecisionFunctionGenerator.h
 */

#ifndef H_RAFL_DECISIONFUNCTIONGENERATOR
#define H_RAFL_DECISIONFUNCTIONGENERATOR

#include <utility>

#include "../base/ProbabilityMassFunction.h"
#include "../examples/Example.h"
#include "DecisionFunction.h"

namespace rafl {

/**
 * \brief An instance of an instantiation of this class template can be used to pick an appropriate decision function to split a set of examples.
 */
template <typename Label>
class DecisionFunctionGenerator
{
  //#################### TYPEDEFS ####################
protected:
  typedef boost::shared_ptr<const Example<Label> > Example_CPtr;

  //#################### DESTRUCTOR ####################
public:
  /**
   * \brief Destroys the generator.
   */
  virtual ~DecisionFunctionGenerator() {}

  //#################### PRIVATE ABSTRACT MEMBER FUNCTIONS ####################
private:
  /**
   * \brief Generates a candidate decision function to split the specified set of examples.
   *
   * \param examples  The examples to split.
   * \return          The candidate decision function.
   */
  virtual DecisionFunction_Ptr generate_candidate(const std::vector<Example_CPtr>& examples) const = 0;

  //#################### PUBLIC MEMBER FUNCTIONS ####################
public:
  /**
   * \brief Picks an appropriate decision function to split the specified set of examples.
   *
   * \param examples        The examples to split.
   * \param candidateCount  The number of candidates to evaluate.
   * \return                The chosen decision function.
   */
  DecisionFunction_Ptr pick_decision_function(const std::vector<Example_CPtr>& examples, int candidateCount = 5) const
  {
    float initialEntropy = calculate_entropy(examples);
    std::multimap<float,DecisionFunction_Ptr,std::greater<float> > gainToCandidateMap;

    for(int i = 0; i < candidateCount; ++i)
    {
      // Generate a candidate decision function.
      DecisionFunction_Ptr candidate = generate_candidate(examples);

      // Partition the examples using the candidate.
      std::pair<std::vector<Example_CPtr>,std::vector<Example_CPtr> > examplesPartition;
      for(size_t j = 0, size = examples.size(); j < size; ++j)
      {
        if(candidate->classify_descriptor(*examples[j]->get_descriptor()) == DecisionFunction::DC_LEFT)
        {
          examplesPartition.first.push_back(examples[j]);
        }
        else
        {
          examplesPartition.second.push_back(examples[j]);
        }
      }

      // Calculate the information gain we would obtain by splitting using this candidate.
      float gain = calculate_information_gain(examples, initialEntropy, examplesPartition.first, examplesPartition.second);

      // Add the result to the gain -> candidate map so as to allow us to find a candidate with maximum gain.
      gainToCandidateMap.insert(std::make_pair(gain, candidate));
    }

    // Return a decision function that had maximum gain.
    return gainToCandidateMap.begin()->second;
  }

  //#################### PRIVATE STATIC MEMBER FUNCTIONS ####################
private:
  /**
   * \brief Calculates the entropy of the label distribution of a set of examples.
   *
   * \param examples  The examples for whose label distribution we want to calculate the entropy.
   * \return          The entropy of the examples' label distribution.
   */
  static float calculate_entropy(const std::vector<Example_CPtr>& examples)
  {
    Histogram<Label> histogram;
    for(typename std::vector<Example_CPtr>::const_iterator it = examples.begin(), iend = examples.end(); it != iend; ++it)
    {
      histogram.add((*it)->get_label());
    }
    return ProbabilityMassFunction<Label>(histogram).calculate_entropy();
  }

  /**
   * \brief Calculates the information gain that results from splitting an example set in a particular way.
   *
   * \param examples        The example set.
   * \param initialEntropy  The entropy of the example set before the split.
   * \param leftExamples    The examples that end up in the left half of the split.
   * \param rightExamples   The examples that end up in the right half of the split.
   * \return                The information gain resulting from the split.
   */
  static float calculate_information_gain(const std::vector<Example_CPtr>& examples, float initialEntropy, const std::vector<Example_CPtr>& leftExamples, const std::vector<Example_CPtr>& rightExamples)
  {
    float exampleCount = static_cast<float>(examples.size());
    float leftEntropy = calculate_entropy(leftExamples);
    float rightEntropy = calculate_entropy(rightExamples);
    float leftWeight = leftExamples.size() / exampleCount;
    float rightWeight = rightExamples.size() / exampleCount;
    return initialEntropy - (leftWeight * leftEntropy + rightWeight * rightEntropy);
  }
};

}

#endif
