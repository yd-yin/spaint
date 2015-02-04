/**
 * evaluation: CrossValidationSplitGenerator.cpp
 */

#include "splitgenerators/CrossValidationSplitGenerator.h"
#include "splitgenerators/RNGFunctor.h"

#include <set>
#include <stdexcept>

#include <tvgutil/LimitedContainer.h>

namespace evaluation {

//#################### CONSTRUCTORS ####################

CrossValidationSplitGenerator::CrossValidationSplitGenerator(unsigned int seed, size_t foldCount)
: SplitGenerator(seed), m_foldCount(foldCount)
{}

//#################### PUBLIC MEMBER FUNCTIONS ####################

std::vector<SplitGenerator::Split> CrossValidationSplitGenerator::generate_splits(size_t exampleCount)
{
  if(m_foldCount > exampleCount)
  {
    throw std::runtime_error("Too few examples to divide into the specified number of folds");
  }

  // Allocate the examples to random folds.
  std::vector<int> foldOfExample(exampleCount);
  for(size_t i = 0; i < exampleCount; ++i)
  {
    foldOfExample[i] = static_cast<int>(i % m_foldCount);
  }

  RNGFunctor rngFunctor(m_rng);
  std::random_shuffle(foldOfExample.begin(), foldOfExample.end(), rngFunctor);

#if 1
  std::cout << "foldOfExample: \n" << tvgutil::make_limited_container(foldOfExample, 20) << '\n';
#endif

  // Generate a split for each fold, in which the first set contains all examples except those in the fold,
  // and the second set contains the examples in the fold. The idea is that the non-fold examples will be
  // used for training and the fold examples will be used for validation.
  std::vector<Split> splits;
  for(size_t validationFold = 0; validationFold < m_foldCount; ++validationFold)
  {
    // Split the examples into two halves.
    Split split;
    for(size_t i = 0; i < exampleCount; ++i)
    {
      (foldOfExample[i] == validationFold ? split.second : split.first).push_back(i);
    }

    // Randomly shuffle the indices in each half of the split.
    std::random_shuffle(split.first.begin(), split.first.end(), rngFunctor);
    std::random_shuffle(split.second.begin(), split.second.end(), rngFunctor);

#if 1
    std::cout << "Fold: " << validationFold << "\n";
    std::cout << "First: \n" << tvgutil::make_limited_container(split.first, 20) << "\n";
    std::cout << "Second: \n" << tvgutil::make_limited_container(split.second, 20) << "\n\n";
#endif

    splits.push_back(split);
  }

  return splits;
}

}
