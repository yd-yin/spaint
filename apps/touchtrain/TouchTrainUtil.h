/**
 * touchtrain: TouchTrainUtil.h
 */

#ifndef H_TOUCHTRAINUTIL
#define H_TOUCHTRAINUTIL

#include <fstream>
#include <string>

#include <boost/lexical_cast.hpp>
#include <boost/tokenizer.hpp>

#include <rafl/examples/Example.h>

#include <spaint/touch/TouchUtil.h>

/**
 * \brief A struct that represents a labelled image path.
 */
template <typename Label>
struct LabelledImagePath
{
  //#################### PUBLIC VARIABLES ####################

  /** The path to an image. */
  std::string m_imagePath;

  /** The label associated with the image. */
  Label m_label;

  //#################### CONSTRUCTORS ####################

  /**
   * \brief Constructs the labelled image path.
   *
   * \param imagePath  The path the the image.
   * \param label      The label associated with the specified image.
   */
  LabelledImagePath(const std::string& imagePath, const Label& label)
  : m_imagePath(imagePath), m_label(label)
  {}
};

/**
 * \brief This class contains functions that help us to load training annotation and generate examples from the annotation.
 */
struct TouchTrainUtil
{
  //#################### PUBLIC STATIC MEMBER FUNCTIONS ####################

  /**
   * \brief Generates an array of instance data in the following format: <path-to-image,label>
   *
   * \param imagePath       The path to the specified images directory.
   * \param annotationPath  The path to the specified file containing the associated annotation.
   *
   * The annotation is assumed to be in the following format: <imageName,label>
   *
   * \return  An array containing the instance data.
   */
  template <typename Label>
  static std::vector<LabelledImagePath<Label> > load_instances(const std::string& imagePath, const std::string& annotationPath)
  {
    std::vector<LabelledImagePath<Label> > instances;

    std::ifstream fs(annotationPath.c_str());
    std::string line;
    while(std::getline(fs, line))
    {
      typedef boost::char_separator<char> sep;
      typedef boost::tokenizer<sep> tokenizer;
      tokenizer tok(line.begin(), line.end(), sep(", \r"));
      std::vector<std::string> tokens(tok.begin(), tok.end());

      std::string imageName = tokens[0];
      Label label = boost::lexical_cast<Label>(tokens.back());
      instances.push_back(LabelledImagePath<Label>(imagePath + "/" + imageName, label));
    }

    return instances;
  }

  /**
   * \brief Generates an array of examples given an array of instance data.
   *
   * \param instances  The instance data.
   * \return           The examples.
   */
  template <typename Label>
  static std::vector<boost::shared_ptr<const Example<Label> > > generate_examples(const std::vector<std::vector<LabelledImagePath<Label> > >& instances)
  {
    typedef boost::shared_ptr<const Example<Label> > Example_CPtr;
    std::vector<Example_CPtr> result;

    for(size_t i = 0, iend = instances.size(); i < iend; ++i)
    {
      for(size_t j = 0, jend = instances[i].size(); j < jend; ++j)
      {
#if 0
        std::cout << "Filename: " << instances[i][j].first << " Label: " << instances[i][j].second << std::endl;
#endif

        af::array img = af::loadImage(instances[i][j].m_imagePath.c_str());
        rafl::Descriptor_CPtr descriptor = spaint::TouchUtil::extract_touch_feature(img);
        result.push_back(Example_CPtr(new Example<Label>(descriptor, instances[i][j].m_label)));
      }
    }

    return result;
  }
};

#endif
