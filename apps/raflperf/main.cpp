/**
 * raflperf: main.cpp
 */

#include <cmath>

#include <boost/assign/list_of.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
using boost::assign::list_of;
using boost::assign::map_list_of;

#include <evaluation/core/PerformanceTable.h>
#include <evaluation/splitgenerators/CrossValidationSplitGenerator.h>
#include <evaluation/splitgenerators/RandomlyPermuteAndDivideSplitGenerator.h>
#include <evaluation/util/CartesianProductParameterSetGenerator.h>
using namespace evaluation;

#include <rafl/examples/ExampleUtil.h>
#include <rafl/examples/UnitCircleExampleGenerator.h>
using namespace rafl;

#include "RandomForestEvaluator.h"

typedef int Label;
typedef boost::shared_ptr<const Example<Label> > Example_CPtr;

typedef CartesianProductParameterSetGenerator::ParamSet ParamSet;

std::string get_iso_timestamp()
{
  boost::posix_time::ptime currentDateTime(boost::posix_time::second_clock::local_time());
  return boost::posix_time::to_iso_string(currentDateTime);
}

int main(int argc, char *argv[])
{
  if(argc != 1 && argc != 4)
  {
    std::cerr << "Usage: raflperf [<training set file> <test set file> <output path>]\n";
    return EXIT_FAILURE;
  }

  std::vector<Example_CPtr> examples;
  std::string outputResultPath;

  if(argc == 1)
  {
    //Generate a set of labels.
    std::set<Label> classLabels;
    classLabels.insert(1);
    classLabels.insert(3);
    classLabels.insert(5);
    classLabels.insert(7);

    //Generate examples around the unit circle.
    UnitCircleExampleGenerator<Label> uceg(classLabels, 1234);
    examples = uceg.generate_examples(classLabels, 100);
    outputResultPath = "UnitCircleExampleGenerator-Results.txt";
  }
  else if(argc == 4)
  {
    std::string trainingSetPath = argv[1];
    std::string testingSetPath = argv[2];
    outputResultPath = argv[3];

    std::cout << "Training set: " << trainingSetPath << "\n";
    std::cout << "Testing set: " << testingSetPath << "\n";

    examples = ExampleUtil::load_examples<Label>(trainingSetPath);
    std::vector<Example_CPtr> testingExamples = ExampleUtil::load_examples<Label>(testingSetPath);

    examples.insert(examples.end(), testingExamples.begin(), testingExamples.end() );
    std::cout << "number of examples= " << examples.size() << "\n";
  }

  //Generate parameters of your algorithm.
  std::vector<ParamSet> params = CartesianProductParameterSetGenerator()
    .add_param("treeCount", list_of<size_t>(1)(5)(9)(13))
    .add_param("splitBudget", list_of<size_t>(static_cast<size_t>(pow(2,10)))(static_cast<size_t>(pow(2,20))))
    .add_param("candidateCount", list_of<int>(256))
    .add_param("decisionFunctionGeneratorType", list_of<std::string>("FeatureThresholding"))
    .add_param("gainThreshold", list_of<float>(0.0f))
    .add_param("maxClassSize", list_of<size_t>(10000))
    .add_param("maxTreeHeight", list_of<size_t>(20))
    .add_param("randomSeed", list_of<unsigned int>(1234))
    .add_param("seenExamplesThreshold", list_of<size_t>(50))
    .add_param("splittabilityThreshold", list_of<float>(0.5f))
    .generate_param_sets();

  const size_t numFolds = 2;
  const unsigned int seed = 1234;

  PerformanceTable results(list_of("Accuracy"));
  //SplitGenerator_Ptr splitGenerator(new CrossValidationSplitGenerator(seed, numFolds));
  SplitGenerator_Ptr splitGenerator(new RandomlyPermuteAndDivideSplitGenerator(seed, 5, 0.5f));
  boost::shared_ptr<RandomForestEvaluator<Label> > evaluator;
  for(size_t n = 0, nend = params.size(); n < nend; ++n)
  {
    evaluator.reset(new RandomForestEvaluator<Label>(splitGenerator, params[n]));
    std::map<std::string,PerformanceMeasure> result = evaluator->evaluate(examples);
    results.record_performance(params[n], result);
  }

  //Output results to the screen.
  results.output(std::cout);

  //Time-stamp the results file.
  outputResultPath += "-" + get_iso_timestamp();

  //Output results to a file.
  std::ofstream resultsFile(outputResultPath.c_str());
  if(!resultsFile)
  {
    std::cout << "Warning could not open file for writing..\n";
  }
  else
  {
    results.output(resultsFile);
  }

  return 0;
}

