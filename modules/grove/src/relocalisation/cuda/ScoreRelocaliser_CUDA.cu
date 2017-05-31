/**
 * grove: ScoreRelocaliser_CUDA.cu
 * Copyright (c) Torr Vision Group, University of Oxford, 2017. All rights reserved.
 */

#include "relocalisation/cuda/ScoreRelocaliser_CUDA.h"

#include <ITMLib/Engines/LowLevel/ITMLowLevelEngineFactory.h>
#include <ITMLib/Utils/ITMLibSettings.h>
using namespace ITMLib;

#include <itmx/base/MemoryBlockFactory.h>
using itmx::MemoryBlockFactory;

#include "clustering/ExampleClustererFactory.h"
#include "features/FeatureCalculatorFactory.h"
#include "forests/DecisionForestFactory.h"
#include "ransac/RansacFactory.h"
#include "relocalisation/shared/ScoreRelocaliser_Shared.h"
#include "reservoirs/ExampleReservoirsFactory.h"

using namespace tvgutil;

namespace grove {

//#################### CUDA KERNELS ####################

template <int TREE_COUNT>
__global__ void ck_score_relocaliser_get_predictions(const ScorePrediction *leafPredictions,
                                                     const ORUtils::VectorX<int, TREE_COUNT> *leafIndices,
                                                     ScorePrediction *outPredictions,
                                                     Vector2i imgSize,
                                                     int nbMaxPredictions)
{
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= imgSize.x || y >= imgSize.y) return;

  get_prediction_for_leaf_shared(leafPredictions, leafIndices, outPredictions, imgSize, nbMaxPredictions, x, y);
}

//#################### CONSTRUCTORS ####################

ScoreRelocaliser_CUDA::ScoreRelocaliser_CUDA(const SettingsContainer_CPtr& settings, const std::string& forestFilename)
  : ScoreRelocaliser(settings, forestFilename)
{
  // Instantiate the sub-algorithms knowing that we are running on the GPU.

  // Features.
  m_featureCalculator = FeatureCalculatorFactory::make_da_rgbd_patch_feature_calculator(ITMLibSettings::DEVICE_CUDA);

  // LowLevelEngine.
  m_lowLevelEngine.reset(ITMLowLevelEngineFactory::MakeLowLevelEngine(ITMLibSettings::DEVICE_CUDA));

  // Forest.
  m_scoreForest = DecisionForestFactory<DescriptorType, FOREST_TREE_COUNT>::make_forest(ITMLibSettings::DEVICE_CUDA,
                                                                                        m_forestFilename);

  // These variables have to be set here, since they depend on the forest that has just been loaded.
  m_reservoirsCount = m_scoreForest->get_nb_leaves();
  m_predictionsBlock = MemoryBlockFactory::instance().make_block<ScorePrediction>(m_reservoirsCount);

  // Reservoirs.
  m_exampleReservoirs = ExampleReservoirsFactory<ExampleType>::make_reservoirs(
      ITMLibSettings::DEVICE_CUDA, m_reservoirCapacity, m_reservoirsCount, m_rngSeed);

  // Clustering.
  m_exampleClusterer = ExampleClustererFactory<ExampleType, ClusterType, PredictionType::MAX_CLUSTERS>::make_clusterer(
      ITMLibSettings::DEVICE_CUDA, m_clustererSigma, m_clustererTau, m_maxClusterCount, m_minClusterSize);

  // P-RANSAC.
  m_preemptiveRansac = RansacFactory::make_preemptive_ransac(ITMLibSettings::DEVICE_CUDA, m_settings);

  // Clear internal state.
  reset();
}

//#################### PUBLIC VIRTUAL MEMBER FUNCTIONS ####################

ScorePrediction ScoreRelocaliser_CUDA::get_raw_prediction(uint32_t treeIdx, uint32_t leafIdx) const
{
  if (treeIdx >= m_scoreForest->get_nb_trees() || leafIdx >= m_scoreForest->get_nb_leaves_in_tree(treeIdx))
  {
    throw std::invalid_argument("Invalid tree or leaf index.");
  }

  return m_predictionsBlock->GetElement(leafIdx * m_scoreForest->get_nb_trees() + treeIdx, MEMORYDEVICE_CUDA);
}

//#################### PROTECTED VIRTUAL MEMBER FUNCTIONS ####################

void ScoreRelocaliser_CUDA::get_predictions_for_leaves(const LeafIndicesImage_CPtr &leafIndices,
                                                       const ScorePredictionsBlock_CPtr &leafPredictions,
                                                       ScorePredictionsImage_Ptr &outputPredictions) const
{
  const Vector2i imgSize = leafIndices->noDims;
  const LeafIndices *leafIndicesData = leafIndices->GetData(MEMORYDEVICE_CUDA);

  // Leaf predictions
  const ScorePrediction *leafPredictionsData = leafPredictions->GetData(MEMORYDEVICE_CUDA);

  // NOP after the first time.
  outputPredictions->ChangeDims(imgSize);
  ScorePrediction *outPredictionsData = outputPredictions->GetData(MEMORYDEVICE_CUDA);

  const dim3 blockSize(32, 32);
  const dim3 gridSize((imgSize.x + blockSize.x - 1) / blockSize.x, (imgSize.y + blockSize.y - 1) / blockSize.y);

  ck_score_relocaliser_get_predictions<<<gridSize, blockSize>>>(
      leafPredictionsData, leafIndicesData, outPredictionsData, imgSize, m_maxClusterCount);
  ORcudaKernelCheck;
}

} // namespace grove
