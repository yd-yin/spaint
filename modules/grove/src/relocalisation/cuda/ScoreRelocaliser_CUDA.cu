/**
 * grove: ScoreRelocaliser_CUDA.cu
 * Copyright (c) Torr Vision Group, University of Oxford, 2017. All rights reserved.
 */

#include "relocalisation/cuda/ScoreRelocaliser_CUDA.h"

#include <itmx/base/MemoryBlockFactory.h>
using itmx::MemoryBlockFactory;

#include "relocalisation/shared/ScoreRelocaliser_Shared.h"
#include "reservoirs/ExampleReservoirsFactory.h"

using namespace tvgutil;

namespace grove {

//#################### CUDA KERNELS ####################

template <int TREE_COUNT>
__global__ void ck_score_relocaliser_get_predictions(const ScorePrediction *leafPredictions, const ORUtils::VectorX<int, TREE_COUNT> *leafIndices,
                                                     ScorePrediction *outPredictions, Vector2i imgSize, int nbMaxPredictions)
{
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  if(x >= imgSize.x || y >= imgSize.y) return;

  get_prediction_for_leaf_shared(leafPredictions, leafIndices, outPredictions, imgSize, nbMaxPredictions, x, y);
}

//#################### CONSTRUCTORS ####################

ScoreRelocaliser_CUDA::ScoreRelocaliser_CUDA(const SettingsContainer_CPtr& settings, const std::string& forestFilename)
: ScoreRelocaliser(settings, DEVICE_CUDA, forestFilename)
{}

//#################### PUBLIC VIRTUAL MEMBER FUNCTIONS ####################

ScorePrediction ScoreRelocaliser_CUDA::get_raw_prediction(uint32_t treeIdx, uint32_t leafIdx) const
{
  if(treeIdx >= m_scoreForest->get_nb_trees() || leafIdx >= m_scoreForest->get_nb_leaves_in_tree(treeIdx))
  {
    throw std::invalid_argument("Invalid tree or leaf index.");
  }

  return m_relocaliserState->predictionsBlock->GetElement(leafIdx * m_scoreForest->get_nb_trees() + treeIdx, MEMORYDEVICE_CUDA);
}

std::vector<Keypoint3DColour> ScoreRelocaliser_CUDA::get_reservoir_contents(uint32_t treeIdx, uint32_t leafIdx) const
{
  if(treeIdx >= m_scoreForest->get_nb_trees() || leafIdx >= m_scoreForest->get_nb_leaves_in_tree(treeIdx))
  {
    throw std::invalid_argument("Invalid tree or leaf index.");
  }

  const uint32_t linearReservoirIdx = leafIdx * m_scoreForest->get_nb_trees() + treeIdx;
  const uint32_t currentReservoirSize = m_relocaliserState->exampleReservoirs->get_reservoir_sizes()->GetElement(linearReservoirIdx, MEMORYDEVICE_CUDA);
  const uint32_t reservoirCapacity = m_relocaliserState->exampleReservoirs->get_reservoir_capacity();

  std::vector<Keypoint3DColour> reservoirContents;
  reservoirContents.reserve(currentReservoirSize);

  m_relocaliserState->exampleReservoirs->get_reservoirs()->UpdateHostFromDevice();
  const Keypoint3DColour *reservoirData = m_relocaliserState->exampleReservoirs->get_reservoirs()->GetData(MEMORYDEVICE_CPU);

  for(uint32_t i = 0; i < currentReservoirSize; ++i)
  {
    reservoirContents.push_back(reservoirData[linearReservoirIdx * reservoirCapacity + i]);
  }

  return reservoirContents;
}

//#################### PROTECTED VIRTUAL MEMBER FUNCTIONS ####################

void ScoreRelocaliser_CUDA::get_predictions_for_leaves(const LeafIndicesImage_CPtr& leafIndices, const ScorePredictionsMemoryBlock_CPtr& leafPredictions,
                                                       ScorePredictionsImage_Ptr& outputPredictions) const
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
    leafPredictionsData, leafIndicesData, outPredictionsData, imgSize, m_maxClusterCount
  );
  ORcudaKernelCheck;
}

}
