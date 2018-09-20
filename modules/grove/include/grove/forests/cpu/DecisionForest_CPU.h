/**
 * grove: DecisionForest_CPU.h
 * Copyright (c) Torr Vision Group, University of Oxford, 2017. All rights reserved.
 */

#ifndef H_GROVE_DECISIONFOREST_CPU
#define H_GROVE_DECISIONFOREST_CPU

#include "../interface/DecisionForest.h"

namespace grove {

/**
 * \brief An instance of this class represents a binary decision forest composed of a fixed number of trees.
 *        The evaluation is performed on the CPU.
 *
 * \note  Training is not performed by this class. We use the node indexing technique described in:
 *        "Implementing Decision Trees and Forests on a GPU" (Toby Sharp, 2008).
 *
 * \tparam DescriptorType The type of descriptor used to find the leaves. Must have a floating-point member array named "data".
 * \tparam TreeCount      The number of trees in the forest. Fixed at compilation time to allow the definition of a data type
 *                        representing the leaf indices.
 */
template <typename DescriptorType, int TreeCount>
class DecisionForest_CPU : public DecisionForest<DescriptorType,TreeCount>
{
  //#################### TYPEDEFS AND USINGS ####################
public:
  typedef DecisionForest<DescriptorType,TreeCount> Base;

  using Base::TREE_COUNT;
  using typename Base::DescriptorImage;
  using typename Base::DescriptorImage_Ptr;
  using typename Base::DescriptorImage_CPtr;
  using typename Base::LeafIndices;
  using typename Base::LeafIndicesImage;
  using typename Base::LeafIndicesImage_Ptr;
  using typename Base::LeafIndicesImage_CPtr;
  using typename Base::NodeEntry;

  //#################### CONSTRUCTORS ####################
public:
  /**
   * \brief Loads the branching structure of a pre-trained decision forest from a file on disk.
   *
   * \param filename The path to the file containing the forest.
   *
   * \throws std::runtime_error If the forest cannot be loaded.
   */
  explicit DecisionForest_CPU(const std::string& filename);

  /** Override */
  explicit DecisionForest_CPU(const tvgutil::SettingsContainer_CPtr& settings);

#ifdef WITH_SCOREFORESTS
  /**
   * \brief Constructs a decision forest by converting an EnsembleLearner that was pre-trained using ScoreForests.
   *
   * \param pretrainedForest The pre-trained forest to convert.
   *
   * \throws std::runtime_error If the pre-trained forest cannot be converted.
   */
  explicit DecisionForest_CPU(const EnsembleLearner& pretrainedForest);
#endif

  //#################### PUBLIC MEMBER FUNCTIONS ####################
public:
  /** Override */
  virtual void find_leaves(const DescriptorImage_CPtr& descriptors, LeafIndicesImage_Ptr& leafIndices) const;
};

}

#endif
