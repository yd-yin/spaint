/**
 * touchtrain: main.cpp
 * Copyright (c) Torr Vision Group, University of Oxford, 2015. All rights reserved.
 */

#include "LabelledPath.h"

#include <boost/assign/list_of.hpp>
#include <boost/filesystem.hpp>
#include <boost/format.hpp>
using boost::assign::list_of;
using boost::assign::map_list_of;

#include <evaluation/core/PerformanceTable.h>
#include <evaluation/splitgenerators/CrossValidationSplitGenerator.h>
#include <evaluation/splitgenerators/RandomPermutationAndDivisionSplitGenerator.h>
#include <evaluation/util/CartesianProductParameterSetGenerator.h>
using namespace evaluation;

#include <rafl/examples/ExampleUtil.h>
#include <rafl/examples/UnitCircleExampleGenerator.h>
using namespace rafl;

#include <raflevaluation/RandomForestEvaluator.h>
using namespace raflevaluation;

#include <spaint/touch/TouchDescriptorCalculator.h>
using namespace spaint;

#include <tvgutil/SerializationUtil.h>
#include <tvgutil/timing/Timer.h>
#include <tvgutil/timing/TimeUtil.h>
using namespace tvgutil;

//#################### TYPEDEFS ####################

typedef int Label;
typedef DecisionTree<Label> DT;
typedef boost::shared_ptr<const Example<Label> > Example_CPtr;
typedef CartesianProductParameterSetGenerator::ParamSet ParamSet;
typedef RandomForest<Label> RF;
typedef boost::shared_ptr<RF> RF_Ptr;

//#################### FUNCTIONS ####################

/**
 * \brief Checks whether a specified path exists.
 *
 * If the path is not found, it outputs the expected path to std::cout.
 *
 * \param path  The path.
 * \return      True, if the path exists, false otherwise.
 */
bool check_path_exists(const std::string& path)
{
  if(!boost::filesystem::exists(path))
  {
    std::cout << "[touchtrain] Expecting to see: " << path << std::endl;
    return false;
  }
  else
  {
    return true;
  }
}

/**
 * \brief Generates an array of examples given an array of labelled image paths.
 *
 * \param labelledImagePaths  The array of labelled image paths.
 * \return                    The examples.
 */
template <typename Label>
static std::vector<boost::shared_ptr<const rafl::Example<Label> > > generate_examples(const std::vector<LabelledPath<Label> >& labelledImagePaths)
{
  typedef boost::shared_ptr<const rafl::Example<Label> > Example_CPtr;
  int labelledImagePathCount = static_cast<int>(labelledImagePaths.size());
  std::vector<Example_CPtr> result(labelledImagePathCount);

  for(int i = 0; i < labelledImagePathCount; ++i)
  {
      af::array img = af::loadImage(labelledImagePaths[i].path.c_str());
      rafl::Descriptor_CPtr descriptor = TouchDescriptorCalculator::calculate_histogram_descriptor(img);
      result[i].reset(new rafl::Example<Label>(descriptor, labelledImagePaths[i].label));
  }

  return result;
}

/**
 * \brief Generates a labelled path for each image in the specified images directory.
 *        The labels for the various images are supplied in a separate annotation file.
 *
 * \param imagesPath      The path to the images directory.
 * \param annotationPath  The path to a file containing the labels to associate with the images in the images path.
 *
 * The annotation is assumed to be in the following format: <imageName,label>
 *
 * \return   The labelled paths for all images in the specified images directory.
 */
template <typename Label>
static std::vector<LabelledPath<Label> > generate_labelled_image_paths(const std::string& imagesPath, const std::string& annotationPath)
{
  // FIXME: Make this robust to bad data.

  std::vector<LabelledPath<Label> > labelledImagePaths;

  std::ifstream fs(annotationPath.c_str());
  if(!fs) throw std::runtime_error("The file: " + annotationPath + " could not be opened.");

  const std::string delimiters(", \r");
  std::vector<std::vector<std::string> > wordLines = WordExtractor::extract_word_lines(fs, delimiters);

  for(size_t i = 0, lineCount = wordLines.size(); i < lineCount; ++i)
  {
    const std::vector<std::string>& words = wordLines[i];
    const std::string& imageFilename = words[0];
    Label label = boost::lexical_cast<Label>(words.back());
    labelledImagePaths.push_back(LabelledPath<Label>(imagesPath + "/" + imageFilename, label));
  }

  return labelledImagePaths;
}

/**
 * \brief A struct that represents the file structure for the touch training data.
 */
struct TouchTrainData
{
  //#################### TYPEDEFS ####################

  typedef std::vector<LabelledPath<Label> > LabelledImagePaths;

  //#################### PUBLIC VARIABLES ####################

  /** The directory containing tables of results generated during cross-validation. */
  std::string m_crossValidationResults;

  /** An array of labelled image paths. */
  LabelledImagePaths m_labelledImagePaths;

  /** The directory where the random forest models are stored. */
  std::string m_models;

  /** The root directory in which the touch training data is stored. */
  std::string m_root;

  //#################### CONSTRUCTORS ####################

  /**
   * \brief Constructs the paths and data relevant for touch training.
   *
   * \param root             The root directory containing the touch training data.
   * \param sequenceNumbers  An array containing the sequence numbers to be included during training.
   */
  TouchTrainData(const std::string& root, std::vector<size_t> sequenceNumbers)
  : m_root(root)
  {
    size_t invalidCount = 0;

    m_crossValidationResults = root + "/crossvalidation-results";
    if(!check_path_exists(m_crossValidationResults)) ++invalidCount;

    m_models = root + "/models";
    if(!check_path_exists(m_models)) ++invalidCount;

    boost::format threeDigits("%03d");
    for(size_t i = 0, size = sequenceNumbers.size(); i < size; ++i)
    {
      std::string sequencePath = root + "/seq" + (threeDigits % sequenceNumbers[i]).str();
      if(!check_path_exists(sequencePath)) ++invalidCount;

      std::string imagePath = sequencePath + "/images";
      std::string annotationPath = sequencePath + "/annotation.txt";
      if(!check_path_exists(imagePath)) ++invalidCount;
      if(!check_path_exists(annotationPath)) ++invalidCount;

      LabelledImagePaths labelledImagePathSet = generate_labelled_image_paths<Label>(imagePath, annotationPath);

      if(labelledImagePathSet.empty())
      {
        std::cout << "[touchtrain] Expecting some data in: " << sequencePath << std::endl;
        ++invalidCount;
      }

      // Append the labelled image paths from the current sequence directory to the global set.
      m_labelledImagePaths.insert(m_labelledImagePaths.end(), labelledImagePathSet.begin(), labelledImagePathSet.end());
    }

    if(invalidCount > 0)
    {
      throw std::runtime_error("The aforementioned directories were not found, please create and populate them.");
    }
  }
};

int main(int argc, char *argv[])
{
#if WITH_OPENMP
  omp_set_nested(1);
#endif

  const unsigned int seed = 12345;

  if(argc != 2)
  {
    std::cerr << "Usage: raflperf [<touch training set path>]\n";
    return EXIT_FAILURE;
  }

  const size_t treeCount = 8;
  const size_t splitBudget = 1048576/2;

  TouchTrainData touchDataset(argv[1],list_of(2)(3)(4)(5));
  std::cout << "[touchtrain] Training set root: " << touchDataset.m_root << '\n';

  std::cout << "[touchtrain] Generating examples...\n";
  std::vector<Example_CPtr> examples = generate_examples<Label>(touchDataset.m_labelledImagePaths);
  std::cout << "[touchtrain] Number of examples = " << examples.size() << '\n';

  // Generate the parameter sets with which to test the random forest.
  std::vector<ParamSet> params = CartesianProductParameterSetGenerator()
    .add_param("treeCount", list_of<size_t>(treeCount))
    .add_param("splitBudget", list_of<size_t>(splitBudget))
    .add_param("candidateCount", list_of<int>(256))
    .add_param("decisionFunctionGeneratorParams", list_of<std::string>(""))
    .add_param("decisionFunctionGeneratorType", list_of<std::string>("FeatureThresholding"))
    .add_param("gainThreshold", list_of<float>(0.0f))
    .add_param("maxClassSize", list_of<size_t>(1000))
    .add_param("maxTreeHeight", list_of<size_t>(20))
    .add_param("randomSeed", list_of<unsigned int>(seed))
    .add_param("seenExamplesThreshold", list_of<size_t>(32)(64)(128))
    .add_param("splittabilityThreshold", list_of<float>(0.3f)(0.5f)(0.8f))
    .add_param("usePMFReweighting", list_of<bool>(false)(true))
    .generate_param_sets();

  // Register the relevant decision function generators with the factory.
  DecisionFunctionGeneratorFactory<Label>::instance().register_rafl_makers();

  // Construct the split generator.
#if 1
  const size_t foldCount = 5;
  SplitGenerator_Ptr splitGenerator(new CrossValidationSplitGenerator(seed, foldCount));
#else
  const size_t splitCount = 5;
  const float ratio = 0.5f;
  SplitGenerator_Ptr splitGenerator(new RandomPermutationAndDivisionSplitGenerator(seed, splitCount, ratio));
#endif

  // Time the random forest.
  Timer<boost::chrono::seconds> timer("ForestEvaluationTime");

  // Evaluate the random forest on the various different parameter sets.
  std::cout << "[touchtrain] Cross-validating the performance of the forest on various parameter sets...\n";
  PerformanceTable results(list_of("Accuracy"));
  boost::shared_ptr<RandomForestEvaluator<Label> > evaluator;
  for(size_t n = 0, size = params.size(); n < size; ++n)
  {
    evaluator.reset(new RandomForestEvaluator<Label>(splitGenerator, params[n]));
    std::map<std::string,PerformanceMeasure> result = evaluator->evaluate(examples);
    results.record_performance(params[n], result);
  }

  // Output the performance table to the screen.
  results.output(std::cout); std::cout << '\n';

  timer.stop();
  std::cout << "[touchtrain] " << timer << '\n';

  // Get a time-stamp for tagging the resulting files.
  const std::string timeStamp = TimeUtil::get_iso_timestamp();

  // Time-stamp the results file.
  std::string textOutputResultPath =  touchDataset.m_crossValidationResults + "/crossvalidationresults-" + timeStamp + ".txt";

  // Output the performance table to the results file.
  std::ofstream resultsFile(textOutputResultPath.c_str());
  if(!resultsFile)
  {
    std::cout << "[touchtrain] Warning could not open file for writing...\n";
  }
  else
  {
    results.output(resultsFile);
  }

  std::cout << "[touchtrain] Training the forest with the best parameters selected during cross-validation...\n";
  ParamSet bestParams = results.find_best_param_set("Accuracy");
  DT::Settings settings(bestParams);
  RF_Ptr randomForest(new RF(treeCount, settings));
  randomForest->add_examples(examples);

  std::cout << "[touchtrain] The final trained forest statistics:\n";
  if(randomForest->train(splitBudget)) randomForest->output_statistics(std::cout);

  std::string forestPath = touchDataset.m_models + "/randomForest-" + timeStamp + ".rf";
  std::cout << "[touchtrain] Saving the forest to: " << forestPath << "\n";
  SerializationUtil::save_text(touchDataset.m_models + "/randomForest-" + timeStamp + ".rf", *randomForest);

  return 0;
}
