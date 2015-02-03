/**
 * evaluation: RandomlyPermuteAndDivideSplitGenerator.h
 */

#ifndef H_EVALUATION_RANDOMLYPERMUTEANDDIVIDESPLITGENERATOR
#define H_EVALUATION_RANDOMLYPERMUTEANDDIVIDESPLITGENERATOR

#include <tvgutil/RandomNumberGenerator.h>

#include "SplitGenerator.h"

namespace evaluation {

/**
 * \brief An instance of this class can be used to generate example splits for learner evaluation.
 *
 * The approach used here first randomly permutes the example indices, then divides them into two sets,
 * the first of which contains a specified percentage of the examples, and the second of which contains
 * the remaining examples. This is repeated for each of the specified number of splits.
 */
class RandomlyPermuteAndDivideSplitGenerator : public SplitGenerator
{
  //#################### PRIVATE VARIABLES ####################
private:
  /** A ratio, namely [the number of examples in the training set] / [the number of examples in the validation set]. */
  float m_ratio;

  /** The number of splits to generate. */
  size_t m_splitCount;

  //#################### CONSTRUCTORS ####################
public:
  /**
   * \brief Constructs a "randomly permute and divide" split generator.
   *
   * \param seed        The seed to use for the random number generator when generating splits.
   * \param splitCount  The number of splits to generate.
   * \param ratio       A ratio, namely [the number of examples in the training set] / [the number of examples in the validation set].
   */
  RandomlyPermuteAndDivideSplitGenerator(unsigned int seed, size_t splitCount, float ratio);

  //#################### PUBLIC MEMBER FUNCTIONS ####################
public:
  /** Override */
  virtual std::vector<Split> generate_splits(size_t exampleCount);

};

}

#endif
