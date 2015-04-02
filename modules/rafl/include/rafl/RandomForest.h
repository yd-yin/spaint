/**
 * rafl: RandomForest.h
 */

#ifndef H_RAFL_RANDOMFOREST
#define H_RAFL_RANDOMFOREST

#include "DecisionTree.h"

namespace rafl {

template <typename Label> class RandomForest;

}

namespace boost { namespace serialization {
template<typename Archive, typename Label> void load_construct_data(Archive& ar, rafl::RandomForest<Label> *randomForest, const unsigned int file_version);
template<typename Archive, typename Label> void save_construct_data(Archive& ar, const rafl::RandomForest<Label> *randomForest, const unsigned int file_version);
}}

namespace rafl {

/**
 * \brief An instance of an instantiation of this class template represents a random forest.
 */
template <typename Label>
class RandomForest
{
  //#################### TYPEDEFS ####################
private:
  typedef boost::shared_ptr<const Example<Label> > Example_CPtr;
  typedef DecisionTree<Label> DT;
  typedef boost::shared_ptr<DT> DT_Ptr;

  //#################### PRIVATE VARIABLES ####################
private:
  /** The decision trees that collectively make up the random forest. */
  std::vector<DT_Ptr> m_trees;

  //#################### CONSTRUCTORS ####################
public:
  /**
   * \brief Constructs a random forest.
   *
   * \param treeCount The number of decision trees to use in the random forest.
   * \param settings  The settings needed to configure the decision trees.
   */
  RandomForest(size_t treeCount, const Settings<Label>& settings)
  {
    for(size_t i = 0; i < treeCount; ++i)
    {
      m_trees.push_back(DT_Ptr(new DT(settings)));
    }
  }

  RandomForest(const std::vector<DT_Ptr>& trees)
  : m_trees(trees)
  {}

  //#################### PUBLIC MEMBER FUNCTIONS ####################
public:
  /**
   * \brief Adds new training examples to the forest.
   *
   * \param examples  The examples to be added.
   */
  void add_examples(const std::vector<Example_CPtr>& examples)
  {
    // Add the new examples to the different trees.
    for(typename std::vector<DT_Ptr>::const_iterator it = m_trees.begin(), iend = m_trees.end(); it != iend; ++it)
    {
      (*it)->add_examples(examples);
    }
  }

  /**
   * \brief Adds new training examples to the forest.
   *
   * \param examples                      A pool of examples that could potentially be added.
   * \param indices                       The indices of the examples in the pool that should be added to the forest.
   * \throws std::out_of_range_exception  If any of the indices are invalid.
   */
  void add_examples(const std::vector<Example_CPtr>& examples, const std::vector<size_t>& indices)
  {
    // Add the new examples to the different trees.
    for(typename std::vector<DT_Ptr>::const_iterator it = m_trees.begin(), iend = m_trees.end(); it != iend; ++it)
    {
      (*it)->add_examples(examples, indices);
    }
  }

  /**
   * \brief Calculates an overall forest PMF for the specified descriptor.
   *
   * This is simply the average of the PMFs for the specified descriptor in the various decision trees.
   *
   * \param descriptor  The descriptor.
   * \return            The PMF.
   */
  ProbabilityMassFunction<Label> calculate_pmf(const Descriptor_CPtr& descriptor) const
  {
    // Sum the masses from the individual tree PMFs for the descriptor.
    std::map<Label,float> masses;
    for(typename std::vector<DT_Ptr>::const_iterator it = m_trees.begin(), iend = m_trees.end(); it != iend; ++it)
    {
      ProbabilityMassFunction<Label> individualPMF = (*it)->lookup_pmf(descriptor);
      const std::map<Label,float>& individualMasses = individualPMF.get_masses();
      for(typename std::map<Label,float>::const_iterator jt = individualMasses.begin(), jend = individualMasses.end(); jt != jend; ++jt)
      {
        masses[jt->first] += jt->second;
      }
    }

    // Create a normalised probability mass function from the summed masses.
    return ProbabilityMassFunction<Label>(masses);
  }

  /**
   * \brief Outputs the random forest to a stream.
   *
   * \param os  The stream to which to output the forest.
   */
  void output(std::ostream& os) const
  {
    for(size_t i = 0, size = m_trees.size(); i < size; ++i)
    {
      os << "Tree " << i << ":\n";
      m_trees[i]->output(os);
      os << '\n';
    }
  }

  /**
   * \brief Predicts a label for the specified descriptor.
   *
   * \param descriptor  The descriptor.
   * \return            The predicted label.
   */
  Label predict(const Descriptor_CPtr& descriptor) const
  {
    return calculate_pmf(descriptor).calculate_best_label();
  }

  /**
   * \brief Trains the forest by splitting a number of suitable nodes in each tree.
   *
   * The number of nodes that are split in each training step is limited to ensure that a step is not overly costly.
   *
   * \param splitBudget The maximum number of nodes per tree that may be split in this training step.
   */
  void train(size_t splitBudget)
  {
    for(typename std::vector<DT_Ptr>::const_iterator it = m_trees.begin(), iend = m_trees.end(); it != iend; ++it)
    {
      (*it)->train(splitBudget);
    }
  }

  //#################### SERIALIZATION #################### 
private:
  friend class boost::serialization::access;
  template<typename Archive>
  void serialize(Archive& ar, const unsigned int version)
  {
    // Intentionally left empty.
  }

  template<typename Archive, typename Dtype>
  friend void boost::serialization::save_construct_data(Archive& ar, const RandomForest<Dtype> *randomForest, const unsigned int file_version);

  template<typename Archive, typename Dtype>
  friend void boost::serialization::load_construct_data(Archive& ar, RandomForest<Dtype> *randomForest, const unsigned int file_version);
};

}

namespace boost { namespace serialization {
template<typename Archive, typename Dtype>
inline void save_construct_data(Archive& ar, const rafl::RandomForest<Dtype> *randomForest, const unsigned int file_version)
{
  std::cout << "<RF<";
  ar << randomForest->m_trees;
}

template<typename Archive, typename Dtype>
inline void load_construct_data(Archive& ar, rafl::RandomForest<Dtype> *randomForest, const unsigned int file_version)
{
  std::cout << ">RF>";
  typedef boost::shared_ptr<rafl::DecisionTree<Dtype> > DT_Ptr;
  std::vector<DT_Ptr> trees;
  ar >> trees;

  ::new(randomForest)rafl::RandomForest<Dtype>(trees);
}
}}

#endif
