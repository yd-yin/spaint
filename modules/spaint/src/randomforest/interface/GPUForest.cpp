/**
 * spaint: GPUForest.cpp
 * Copyright (c) Torr Vision Group, University of Oxford, 2016. All rights reserved.
 */

#include "randomforest/interface/GPUForest.h"

#include <boost/make_shared.hpp>

#include "util/MemoryBlockFactory.h"

namespace spaint
{
GPUForest::GPUForest(const EnsembleLearner &pretrained_forest)
{
  // Convert list of nodes into an appropriate image
  const int nTrees = pretrained_forest.GetNbTrees();
  const int maxNbNodes = pretrained_forest.GetMaxNbNodesInAnyLearner();

  if (nTrees != NTREES)
  {
    throw std::runtime_error(
        "Number of trees in the loaded forest different from the instantiation of GPUForest.");
  }

  // Create texture storing the nodes
  const MemoryBlockFactory &mbf = MemoryBlockFactory::instance();
  m_forestImage = mbf.make_image<GPUForestNode>(Vector2i(nTrees, maxNbNodes));
  m_forestImage->Clear();

  std::cout << "Forest texture has size: " << m_forestImage->noDims
      << std::endl;

  // Fill the nodes
  GPUForestNode *forestData = m_forestImage->GetData(MEMORYDEVICE_CPU);

  for (int treeIdx = 0; treeIdx < nTrees; ++treeIdx)
  {
    const Learner* tree = pretrained_forest.GetTree(treeIdx);
    const int nbNodes = tree->GetNbNodes();

    // We set the first free entry to 1 since we reserve 0 for the root
    int first_free_idx = convert_node(tree, 0, treeIdx, nTrees, 0, 1,
        forestData);
    std::cout << "Converted tree " << treeIdx << ", had " << nbNodes
        << " nodes." << std::endl;
    std::cout << "Total number of leaves: " << m_leafPredictions.size()
        << std::endl;
  }

  m_predictionsBlock = mbf.make_block<GPUForestPrediction>(
      m_leafPredictions.size());
  convert_predictions();
}

GPUForest::~GPUForest()
{
}

int GPUForest::convert_node(const Learner *tree, int node_idx, int tree_idx,
    int n_trees, int output_idx, int first_free_idx, GPUForestNode *gpu_nodes)
{
  const Node* node = tree->GetNode(node_idx);
  GPUForestNode &gpuNode = gpu_nodes[output_idx * n_trees + tree_idx];

  // The assumption is that output_idx is already reserved for the current node
  if (node->IsALeaf())
  {
//    gpuNode.leafIdx = node_idx; // Node index in the original tree, could be used to get the modes from there
    gpuNode.leftChildIdx = -1; // Is a leaf
    gpuNode.featureIdx = 0;
    gpuNode.featureThreshold = 0.f;
    // first_free_idx does not change

    gpuNode.leafIdx = m_leafPredictions.size();

    // Copy the prediction
    // TODO: possibly drop some modes
    const LeafBPDGaussianMean* leafPtr = ToLeafBPDGaussianMean(node);
    if (leafPtr->GetPrediction())
    {
      const PredictionGaussianMean *pred = ToPredictionGaussianMean(
          leafPtr->GetPrediction());
      m_leafPredictions.push_back(*pred);
    }
    else
    {
      // empty prediction
      m_leafPredictions.push_back(PredictionGaussianMean());
    }

  }
  else
  {
    gpuNode.leafIdx = -1; // Not a leaf

    // Reserve 2 slots for the child nodes.
    gpuNode.leftChildIdx = first_free_idx++;
    int rightChildIdx = first_free_idx++; // No need to store it in the texture since it's always leftChildIdx + 1

    const InnerNode *inner_node = ToInnerNode(node);
    std::vector<float> params = inner_node->GetFeature()->GetParameters();

    gpuNode.featureIdx = params[1];
    gpuNode.featureThreshold = params[2];

    first_free_idx = convert_node(tree, node->GetLeftChildIndex(), tree_idx,
        n_trees, gpuNode.leftChildIdx, first_free_idx, gpu_nodes);
    first_free_idx = convert_node(tree, node->GetRightChildIndex(), tree_idx,
        n_trees, rightChildIdx, first_free_idx, gpu_nodes);
  }

  return first_free_idx;
}

//boost::shared_ptr<EnsemblePredictionGaussianMean> GPUForest::get_prediction_for_leaves(
//    const LeafIndices &leaves)
//{
//  boost::shared_ptr<EnsemblePredictionGaussianMean> res = boost::make_shared<
//      EnsemblePredictionGaussianMean>();
//
//  for (int treeIdx = 0; treeIdx < NTREES; ++treeIdx)
//  {
//    PredictionGaussianMean &currentPred = m_leafPredictions[leaves[treeIdx]];
//    const size_t nbModes = currentPred._modes.size();
//
//    res->_modes.reserve(res->_modes.size() + nbModes); // Make some space
//
////    res->_modes.push_back(std::vector<PredictedGaussianMean*>(nbModes));
//    for (size_t modeIdx = 0; modeIdx < nbModes; ++modeIdx)
//    {
//      // Is a vector because [0] is the position and [1] is the colour of the mode
//      std::vector<PredictedGaussianMean> &currentMode =
//          currentPred._modes[modeIdx];
//      const size_t nbComponents = currentMode.size();
//
//      res->_modes.push_back(std::vector<PredictedGaussianMean*>(nbComponents));
//      for (size_t compIdx = 0; compIdx < nbComponents; ++compIdx)
//      {
//        res->_modes.back()[compIdx] = &currentMode[compIdx];
//      }
//    }
//  }
//
//  return res;
//}

boost::shared_ptr<GPUForestPrediction> GPUForest::get_prediction_for_leaves(
    const LeafIndices &leaves)
{
  boost::shared_ptr<GPUForestPrediction> res = boost::make_shared<
      GPUForestPrediction>();
  res->nbModes = 0;

  int treeModeIdx[NTREES];
  memset(treeModeIdx, 0, sizeof(treeModeIdx));

  const GPUForestPrediction *predictionsData = m_predictionsBlock->GetData(
      MEMORYDEVICE_CPU);

  const GPUForestPrediction *leafPredictions[NTREES];
  for (int treeIdx = 0; treeIdx < NTREES; ++treeIdx)
  {
    leafPredictions[treeIdx] = &predictionsData[leaves[treeIdx]];
  }

  // Merge first MAX_MODES from the sorted mode arrays
  while (res->nbModes < GPUForestPrediction::MAX_MODES)
  {
    int bestTreeIdx = 0;
    int bestTreeNbInliers = 0;

    // Find the tree with most inliers
    for (int treeIdx = 0; treeIdx < NTREES; ++treeIdx)
    {
      if (leafPredictions[treeIdx]->nbModes > treeModeIdx[treeIdx]
          && leafPredictions[treeIdx]->modes[treeModeIdx[treeIdx]].nbInliers
              > bestTreeNbInliers)
      {
        bestTreeIdx = treeIdx;
        bestTreeNbInliers =
            leafPredictions[treeIdx]->modes[treeModeIdx[treeIdx]].nbInliers;
      }
    }

    if (bestTreeNbInliers == 0)
    {
      // No more modes
      break;
    }

    // Copy its mode into the output array, increment its index
    res->modes[res->nbModes] =
        leafPredictions[bestTreeIdx]->modes[treeModeIdx[bestTreeIdx]];
    res->nbModes++;
    treeModeIdx[bestTreeIdx]++;
  }

  return res;
}

void GPUForest::convert_predictions()
{
  GPUForestPrediction *gpuPredictions = m_predictionsBlock->GetData(
      MEMORYDEVICE_CPU);

#pragma omp parallel for
  for (size_t leafIdx = 0; leafIdx < m_leafPredictions.size(); ++leafIdx)
  {
    const PredictionGaussianMean &currentPred = m_leafPredictions[leafIdx];

    // copy to sort modes by descending number of inliers so to keep only the best ones
    auto modes = currentPred._modes;
    std::sort(modes.begin(), modes.end(),
        [](const std::vector<PredictedGaussianMean> &a, const std::vector<PredictedGaussianMean> &b)
        { return a[0]._nbPoints > b[0]._nbPoints;});

    GPUForestPrediction &currentTargetPred = gpuPredictions[leafIdx];
    currentTargetPred.nbModes = 0; // Reset modes

    for (size_t modeIdx = 0;
        modeIdx < modes.size()
            && currentTargetPred.nbModes < GPUForestPrediction::MAX_MODES;
        ++modeIdx)
    {
      const auto &mode = modes[modeIdx];
      auto &targetMode = currentTargetPred.modes[currentTargetPred.nbModes++];

      // Not using _meanf and the others because the float variant sometimes seems not set..
      Eigen::Map<Eigen::Vector3f>(targetMode.position.v) = mode[0]._mean.cast<
          float>();
      Eigen::Map<Eigen::Matrix3f>(targetMode.positionInvCovariance.m) =
          mode[0]._inverseCovariance.cast<float>();
      targetMode.determinant = static_cast<float>(mode[0]._determinant);

      // Downcast the colour from float to uchar (loss of precision is acceptable)
      targetMode.colour.x = static_cast<uint8_t>(mode[1]._mean(0));
      targetMode.colour.y = static_cast<uint8_t>(mode[1]._mean(1));
      targetMode.colour.z = static_cast<uint8_t>(mode[1]._mean(2));

      targetMode.nbInliers = mode[0]._nbPoints;
    }
  }
}

int GPUForestPrediction::get_best_mode(const Vector3f &v) const
{
  float energy;
  return get_best_mode_and_energy(v, energy);
}

int GPUForestPrediction::get_best_mode_and_energy(const Vector3f &v,
    float &maxScore) const
{
  static const float exponent = powf(2.0f * M_PI, 3);

  int argmax = -1;
  maxScore = std::numeric_limits<float>::lowest();

  for (int m = 0; m < nbModes; ++m)
  {
    const float nbPts = static_cast<float>(modes[m].nbInliers);
    const Vector3f diff = v - modes[m].position;

    const float normalization = 1.0 / sqrtf(modes[m].determinant * exponent);
    // This is the textbook implementation of Mahalanobis distance
    // Helpers::MahalanobisSquared3x3 used in the original code seems wrong
    const float mahalanobisSq = dot(diff,
        modes[m].positionInvCovariance * diff);
    const float descriptiveStatistics = expf(-0.5f * mahalanobisSq);
    const float evalGaussian = normalization * descriptiveStatistics;
    const float score = nbPts * evalGaussian;

    if (score > maxScore)
    {
      maxScore = score;
      argmax = m;
    }
  }

  return argmax;
}

}
