/**
 * rafl: DecisionFunctionGenerator.h
 */

#ifndef H_RAFL_DECISIONFUNCTIONGENERATOR
#define H_RAFL_DECISIONFUNCTIONGENERATOR

#ifdef WITH_OPENMP
#include <omp.h>
#endif

#include <utility>

#include "../base/ProbabilityMassFunction.h"
#include "../examples/ExampleReservoir.h"
#include "../examples/ExampleUtil.h"
#include "DecisionFunction.h"

namespace rafl {

/**
 * \brief An instance of an instantiation of a class template deriving from this one can be used to generate a decision function
 *        with which to split a set of examples.
 */
template <typename Label>
class DecisionFunctionGenerator
{
  //#################### PROTECTED TYPEDEFS ####################
protected:
  typedef boost::shared_ptr<const Example<Label> > Example_CPtr;

  //#################### NESTED TYPES ####################
public:
  /**
   * \brief An instance of this struct represents a split of a set of examples into two subsets,
   *        based on their classification against a decision function.
   */
  struct Split
  {
    /** The decision function that induced the split. */
    DecisionFunction_Ptr m_decisionFunction;

    /** The examples that were sent left by the decision function. */
    std::vector<Example_CPtr> m_leftExamples;

    /** The examples that were sent right by the decision function. */
    std::vector<Example_CPtr> m_rightExamples;
  };

  //#################### PUBLIC TYPEDEFS ####################
public:
  typedef boost::shared_ptr<Split> Split_Ptr;
  typedef boost::shared_ptr<const Split> Split_CPtr;

  //#################### PRIVATE VERIABLES #################### 
private:
  mutable std::vector<Split> m_splitCandidates;

  //#################### DESTRUCTOR ####################
public:
  /**
   * \brief Destroys the generator.
   */
  virtual ~DecisionFunctionGenerator() {}

  //#################### PUBLIC ABSTRACT MEMBER FUNCTIONS ####################
public:
  /**
   * \brief Gets the type of the decision function generator.
   *
   * \return  The type of the decision function generator.
   */
  virtual std::string get_type() const = 0;

  //#################### PRIVATE ABSTRACT MEMBER FUNCTIONS ####################
private:
  /**
   * \brief Generates a candidate decision function to split the specified set of examples.
   *
   * \param examples  The examples to split.
   * \return          The candidate decision function.
   */
  virtual DecisionFunction_Ptr generate_candidate_decision_function(const std::vector<Example_CPtr>& examples) const = 0;

  //#################### PUBLIC MEMBER FUNCTIONS ####################
public:
  /**
   * \brief Tries to pick an appropriate way in which to split the specified reservoir of examples.
   *
   * \param reservoir             The reservoir of examples to split.
   * \param candidateCount        The number of candidates to evaluate.
   * \param gainThreshold         The minimum information gain that must be obtained from a split to make it worthwhile.
   * \param inverseClassWeights   The (optional) inverses of the L1-normalised class frequencies observed in the training data.
   * \return                      The chosen split, if one was suitable, or NULL otherwise.
   */
  Split_CPtr split_examples(const ExampleReservoir<Label>& reservoir, int candidateCount, float gainThreshold, const boost::optional<std::map<Label,float> >& inverseClassWeights) const
  {
    if(m_splitCandidates.size() != candidateCount) m_splitCandidates.resize(candidateCount);

    float initialEntropy = ExampleUtil::calculate_entropy(*reservoir.get_histogram(), inverseClassWeights);
    float bestGain = -1;
    int bestId = -1;

#if 0
    std::cout << "\nP: " << *reservoir.get_histogram() << ' ' << initialEntropy << '\n';
#endif

    std::vector<Example_CPtr> examples = reservoir.get_examples();

    //std::vector<Split> splitCandidates(candidateCount);
    for(int i = 0; i < candidateCount; ++i)
    {
      // Generate a decision function for the split candidate.
      m_splitCandidates[i].m_decisionFunction = generate_candidate_decision_function(examples);
    }

#ifdef WITH_OPENMP
#pragma omp parallel for
#endif
    for(int i = 0; i < candidateCount; ++i)
    {
#if 0 && WITH_OPENMP 
      int threadId = omp_get_thread_num();
      int nThreads = omp_get_num_threads();
      std::cout << "threadId=" << threadId << " nThreads=" << nThreads << '\n';
#endif

#if 0
      std::cout << *splitCandidate->m_decisionFunction << '\n';
#endif
      m_splitCandidates[i].m_rightExamples.clear();
      m_splitCandidates[i].m_leftExamples.clear();

     // Partition the examples using the decision function.
      for(size_t j = 0, size = examples.size(); j < size; ++j)
      {
        if(m_splitCandidates[i].m_decisionFunction->classify_descriptor(*examples[j]->get_descriptor()) == DecisionFunction::DC_LEFT)
        {
          m_splitCandidates[i].m_leftExamples.push_back(examples[j]);
        }
        else
        {
          m_splitCandidates[i].m_rightExamples.push_back(examples[j]);
        }
      }

      // Calculate the information gain we would obtain from this split.
      float gain = calculate_information_gain(reservoir, initialEntropy, m_splitCandidates[i].m_leftExamples, m_splitCandidates[i].m_rightExamples, inverseClassWeights);

#ifdef WITH_OPENMP
#pragma omp critical
#endif
      {
        if(gain > bestGain)
        {
          if(gain > gainThreshold && !m_splitCandidates[i].m_leftExamples.empty() && !m_splitCandidates[i].m_rightExamples.empty())
          {
            bestGain = gain;
            bestId = i;
          }
        }

      }
      // Add the result to the gain -> candidate map so as to allow us to find a split with maximum gain.
    }

    Split_Ptr bestSplitCandidate;
    if(bestId != -1)
    {
      bestSplitCandidate.reset(new Split(m_splitCandidates[bestId]));
    }

    // Return a split candidate that had maximum gain (note that this may be NULL if no split had a high enough gain).
    return bestSplitCandidate;
  }

  //#################### PRIVATE STATIC MEMBER FUNCTIONS ####################
private:
  /**
   * \brief Calculates the information gain that results from splitting an example reservoir in a particular way.
   *
   * \param reservoir           The reservoir.
   * \param initialEntropy      The entropy of the example set before the split.
   * \param leftExamples        The examples that end up in the left half of the split.
   * \param rightExamples       The examples that end up in the right half of the split.
   * \param inverseClassWeights The (optional) inverses of the L1-normalised class frequencies observed in the training data.
   * \return                    The information gain resulting from the split.
   */
  static float calculate_information_gain(const ExampleReservoir<Label>& reservoir, float initialEntropy, const std::vector<Example_CPtr>& leftExamples, const std::vector<Example_CPtr>& rightExamples, const boost::optional<std::map<Label,float> >& inverseClassWeights)
  {
    float exampleCount = static_cast<float>(reservoir.current_size());
    std::map<Label,float> multipliers = reservoir.get_class_multipliers();
    if(inverseClassWeights) multipliers = combine_multipliers(multipliers, *inverseClassWeights);

    float leftEntropy = ExampleUtil::calculate_entropy(leftExamples, multipliers);
    float rightEntropy = ExampleUtil::calculate_entropy(rightExamples, multipliers);
    float leftWeight = leftExamples.size() / exampleCount;
    float rightWeight = rightExamples.size() / exampleCount;

#if 0
    std::cout << "L: " << ExampleUtil::make_histogram(leftExamples) << ' ' << leftEntropy << '\n';
    std::cout << "R: " << ExampleUtil::make_histogram(rightExamples) << ' ' << rightEntropy << '\n';
#endif

    float gain = initialEntropy - (leftWeight * leftEntropy + rightWeight * rightEntropy);

#if 0
    std::cout << "Gain: " << gain << '\n';
#endif

    return gain;
  }

  /**
   * \brief Multiplies together two sets of multipliers that share some labels in common.
   *
   * Multipliers that only appear in one of the two input sets will not be included in the result,
   * e.g. combine_multipliers({a => 0.1, b => 0.2}, {b => 0.5, c => 0.3}) = {b => 0.2 * 0.5 = 0.1}.
   *
   * \param multipliers1  The first set of multipliers.
   * \param multipliers2  The second set of multipliers.
   * \return              The combined multipliers.
   */
  static std::map<Label,float> combine_multipliers(const std::map<Label,float>& multipliers1, const std::map<Label,float>& multipliers2)
  {
    std::map<Label,float> result;

    typename std::map<Label,float>::const_iterator it = multipliers1.begin(), iend = multipliers1.end(), jt = multipliers2.begin(), jend = multipliers2.end();
    while(it != iend && jt != jend)
    {
      if(it->first == jt->first)
      {
        result.insert(std::make_pair(it->first, it->second * jt->second));
        ++it, ++jt;
      }
      else if(it->first < jt->first) ++it;
      else ++jt;
    }

    return result;
  }
};

}

#endif
