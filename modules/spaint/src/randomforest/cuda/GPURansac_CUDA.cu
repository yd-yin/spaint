/**
 * spaint: GPURansac_CUDA.cu
 * Copyright (c) Torr Vision Group, University of Oxford, 2016. All rights reserved.
 */

#include "randomforest/cuda/GPURansac_CUDA.h"

#include "util/MemoryBlockFactory.h"

#include <thrust/execution_policy.h>
#include <thrust/sort.h>

namespace spaint
{

namespace
{
__global__ void ck_init_random_states(GPURansac_CUDA::RandomState *randomStates,
    uint32_t nbStates, uint32_t seed)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;

  if (idx >= nbStates)
    return;

  curand_init(seed, idx, 0, &randomStates[idx]);
}

__device__ bool generate_candidate(const RGBDPatchFeature *patchFeaturesData,
    const GPUForestPrediction *predictionsData, const Vector2i &imgSize,
    GPURansac_CUDA::RandomState *randomState, PoseCandidate &poseCandidate,
    bool m_useAllModesPerLeafInPoseHypothesisGeneration,
    bool m_checkMinDistanceBetweenSampledModes,
    float m_minDistanceBetweenSampledModes,
    bool m_checkRigidTransformationConstraint,
    float m_translationErrorMaxForCorrectPose)
{
  static const int m_nbPointsForKabschBoostrap = 3;

//
//  std::uniform_int_distribution<int> col_index_generator(0,
//      m_featureImage->noDims.width - 1);
//  std::uniform_int_distribution<int> row_index_generator(0,
//      m_featureImage->noDims.height - 1);
//
//  const RGBDPatchFeature *patchFeaturesData = m_featureImage->GetData(
//      MEMORYDEVICE_CPU);
//  const GPUForestPrediction *predictionsData = m_predictionsImage->GetData(
//      MEMORYDEVICE_CPU);

  bool foundIsometricMapping = false;
  const int maxIterationsOuter = 20;
  int iterationsOuter = 0;

  while (!foundIsometricMapping && iterationsOuter < maxIterationsOuter)
  {
    ++iterationsOuter;

    int selectedPixelCount = 0;
    int selectedPixelX[m_nbPointsForKabschBoostrap];
    int selectedPixelY[m_nbPointsForKabschBoostrap];
    int selectedPixelMode[m_nbPointsForKabschBoostrap];

    static const int maxIterationsInner = 6000;
    int iterationsInner = 0;

    while (selectedPixelCount != m_nbPointsForKabschBoostrap
        && iterationsInner < maxIterationsInner)
    {
      ++iterationsInner;

      const int x = curand(randomState) % (imgSize.width - 1);
      const int y = curand(randomState) % (imgSize.height - 1);
      // The implementation below sometimes generates OOB values (with 0.999999,
      // with 0.999 it works but seems a weird hack)
//      const int x = __float2int_rz(
//          curand_uniform(randomState) * (imgSize.width - 1 + 0.999999f));
//      const int y = __float2int_rz(
//          curand_uniform(randomState) * (imgSize.height - 1 + 0.999999f));
      const int linearFeatureIdx = y * imgSize.width + x;
      const RGBDPatchFeature &selectedFeature =
          patchFeaturesData[linearFeatureIdx];

      if (selectedFeature.position.w < 0.f) // Invalid feature
        continue;

      const GPUForestPrediction &selectedPrediction =
          predictionsData[linearFeatureIdx];

      if (selectedPrediction.nbModes == 0)
        continue;

      int selectedModeIdx = 0;
      if (m_useAllModesPerLeafInPoseHypothesisGeneration)
      {
        selectedModeIdx = curand(randomState)
            % (selectedPrediction.nbModes - 1);
//        selectedModeIdx = __float2int_rz(
//            curand_uniform(randomState)
//                * (selectedPrediction.nbModes - 1 + 0.999f));
      }

      // This is the first pixel, check that the pixel colour corresponds with the selected mode
      if (selectedPixelCount == 0)
      {
        const Vector3u colourDiff = selectedFeature.colour.toVector3().toUChar()
            - selectedPrediction.modes[selectedModeIdx].colour;
        const bool consistentColour = fabsf(colourDiff.x) <= 30.f
            && fabsf(colourDiff.y) <= 30.f && fabsf(colourDiff.z) <= 30.f;

        if (!consistentColour)
          continue;
      }

      // if (false)
      if (m_checkMinDistanceBetweenSampledModes)
      {
        const Vector3f worldPt =
            selectedPrediction.modes[selectedModeIdx].position;

        // Check that this mode is far enough from the other modes
        bool farEnough = true;

        for (int idxOther = 0; idxOther < selectedPixelCount; ++idxOther)
        {
          const int xOther = selectedPixelX[idxOther];
          const int yOther = selectedPixelY[idxOther];
          const int modeIdxOther = selectedPixelMode[idxOther];

          const int linearIdxOther = yOther * imgSize.width + xOther;
          const GPUForestPrediction &predOther = predictionsData[linearIdxOther];

          Vector3f worldPtOther = predOther.modes[modeIdxOther].position;

          float distOther = length(worldPtOther - worldPt);
          if (distOther < m_minDistanceBetweenSampledModes)
          {
            farEnough = false;
            break;
          }
        }

        if (!farEnough)
          continue;
      }

      // isometry?
      //       if (false)
      // if (true)
      if (m_checkRigidTransformationConstraint)
      {
        bool violatesConditions = false;

        for (int m = 0; m < selectedPixelCount && !violatesConditions; ++m)
        {
          const int xFirst = selectedPixelX[m];
          const int yFirst = selectedPixelY[m];
          const int modeIdxFirst = selectedPixelMode[m];
          const int linearIdxOther = yFirst * imgSize.width + xFirst;
          const GPUForestPrediction &predFirst = predictionsData[linearIdxOther];

          const Vector3f worldPtFirst = predFirst.modes[modeIdxFirst].position;
          const Vector3f worldPtCur =
              selectedPrediction.modes[selectedModeIdx].position;

          float distWorld = length(worldPtFirst - worldPtCur);

          const Vector3f localPred =
              patchFeaturesData[linearIdxOther].position.toVector3();
          const Vector3f localCur = selectedFeature.position.toVector3();

          float distLocal = length(localPred - localCur);

          if (distLocal < m_minDistanceBetweenSampledModes)
            violatesConditions = true;

          if (std::abs(distLocal - distWorld)
              > 0.5f * m_translationErrorMaxForCorrectPose)
          {
            violatesConditions = true;
          }
        }

        if (violatesConditions)
          continue;
      }

      selectedPixelX[selectedPixelCount] = x;
      selectedPixelY[selectedPixelCount] = y;
      selectedPixelMode[selectedPixelCount] = selectedModeIdx;
      ++selectedPixelCount;
    }

    //    std::cout << "Inner iterations: " << iterationsInner << std::endl;

    // Reached limit of iterations
    if (selectedPixelCount != m_nbPointsForKabschBoostrap)
      return false;

    // Populate resulting pose (except the actual pose that is computed on the CPU due to Kabsch)
    foundIsometricMapping = true;
    poseCandidate.nbInliers = selectedPixelCount;
    poseCandidate.energy = 0.f;
    poseCandidate.cameraId = -1;

    for (int s = 0; s < selectedPixelCount; ++s)
    {
      const int x = selectedPixelX[s];
      const int y = selectedPixelY[s];
      const int modeIdx = selectedPixelMode[s];
      const int linearIdx = y * imgSize.width + x;

      poseCandidate.inliers[s].linearIdx = linearIdx;
      poseCandidate.inliers[s].modeIdx = modeIdx;
      poseCandidate.inliers[s].energy = 0.f;
    }
  }

  if (iterationsOuter < maxIterationsOuter)
    return true;

  return false;
}

__global__ void ck_generate_pose_candidates(const RGBDPatchFeature *features,
    const GPUForestPrediction *predictions, const Vector2i imgSize,
    GPURansac_CUDA::RandomState *randomStates, PoseCandidates *poseCandidates,
    int maxNbPoseCandidates,
    bool m_useAllModesPerLeafInPoseHypothesisGeneration,
    bool m_checkMinDistanceBetweenSampledModes,
    float m_minDistanceBetweenSampledModes,
    bool m_checkRigidTransformationConstraint,
    float m_translationErrorMaxForCorrectPose)
{
  const int candidateIdx = blockIdx.x * blockDim.x + threadIdx.x;

  // Reset nuber of candidates. Might put in a different kernel for efficiency.
  if (candidateIdx == 0)
    poseCandidates->nbCandidates = 0;

  __syncthreads();

  if (candidateIdx >= maxNbPoseCandidates)
    return;

  GPURansac_CUDA::RandomState *randomState = &randomStates[candidateIdx];

  PoseCandidate candidate;
  candidate.cameraId = candidateIdx;

  bool valid = generate_candidate(features, predictions, imgSize, randomState,
      candidate, m_useAllModesPerLeafInPoseHypothesisGeneration,
      m_checkMinDistanceBetweenSampledModes, m_minDistanceBetweenSampledModes,
      m_checkRigidTransformationConstraint,
      m_translationErrorMaxForCorrectPose);

  if (valid)
  {
    const int candidateIdx = atomicAdd(&poseCandidates->nbCandidates, 1);

    PoseCandidate *candidates = poseCandidates->candidates;
    candidates[candidateIdx] = candidate;
  }
}

__global__ void ck_compute_energies(const RGBDPatchFeature *features,
    const GPUForestPrediction *predictions, PoseCandidates *poseCandidates)
{
  const int tId = threadIdx.x;
  const int threadsPerBlock = blockDim.x;
  const int candidateIdx = blockIdx.x;
  const int nbCandidates = poseCandidates[0].nbCandidates;

  if (candidateIdx >= nbCandidates)
  {
    // Candidate has been trimmed, entire block returns,
    // does not cause troubles with the following __syncthreads()
    return;
  }

  PoseCandidate &currentCandidate = poseCandidates[0].candidates[candidateIdx];

  if (tId == 0)
  {
    currentCandidate.energy = 0.f;
  }

  __syncthreads();

  float localEnergy = 0.f;

  const int nbInliers = currentCandidate.nbInliers;
  for (int inlierIdx = tId; inlierIdx < nbInliers; inlierIdx += threadsPerBlock)
  {
    const int linearIdx = currentCandidate.inliers[inlierIdx].linearIdx;
    const Vector3f localPixel = features[linearIdx].position.toVector3();
    const Vector3f projectedPixel = currentCandidate.cameraPose * localPixel;

    const GPUForestPrediction &pred = predictions[linearIdx];

    // eval individual energy
    float energy;
    int argmax = pred.get_best_mode_and_energy(projectedPixel, energy);

    // Has at least a valid mode
    if (argmax < 0)
    {
      // should have not been inserted in the inlier set
      printf("prediction has no valid modes\n");
      continue;
    }

    if (pred.modes[argmax].nbInliers == 0)
    {
      // the original implementation had a simple continue
      printf("mode has no inliers\n");
      continue;
    }

    energy /= static_cast<float>(pred.nbModes);
    energy /= static_cast<float>(pred.modes[argmax].nbInliers);

    if (energy < 1e-6f)
      energy = 1e-6f;
    energy = -log10f(energy);

    currentCandidate.inliers[inlierIdx].energy = energy;
    currentCandidate.inliers[inlierIdx].modeIdx = argmax;
    localEnergy += energy;
  }

  // Now reduce by shuffling down the local energies
  //(localEnergy for thread 0 in the warp contains the sum for the warp)
  for (int offset = warpSize / 2; offset > 0; offset /= 2)
    localEnergy += __shfl_down(localEnergy, offset);

  // Thread 0 of each warp updates the final energy
  if (threadIdx.x & (warpSize - 1) == 0)
    atomicAdd(&currentCandidate.energy, localEnergy);

  __syncthreads(); // Wait for all threads in the block

  // tId 0 computes the final energy
  if (tId == 0)
    currentCandidate.energy = currentCandidate.energy
        / static_cast<float>(currentCandidate.nbInliers);
}

__global__ void ck_sort_energies(PoseCandidates *poseCandidates)
{
  const int nbCandidates = poseCandidates->nbCandidates;
  PoseCandidate *candidates = poseCandidates->candidates;

  thrust::sort(candidates, candidates + nbCandidates,
      [=](const PoseCandidate &a, const PoseCandidate &b)
      { return a.energy < b.energy;});

  // Uses dynamic parallelism to launch more kernels (Causes OOM error just by adding this code)
//  thrust::sort(thrust::device, candidates, candidates + nbCandidates,
//      [=](const PoseCandidate &a, const PoseCandidate &b)
//      { return a.energy < b.energy;});
}
}

GPURansac_CUDA::GPURansac_CUDA()
{
  MemoryBlockFactory &mbf = MemoryBlockFactory::instance();
  m_randomStates = mbf.make_block<RandomState>(PoseCandidates::MAX_CANDIDATES);
  m_rngSeed = 42;

  init_random();
}

void GPURansac_CUDA::init_random()
{
  RandomState *randomStates = m_randomStates->GetData(MEMORYDEVICE_CUDA);

  // Initialize random states
  dim3 blockSize(256);
  dim3 gridSize(
      (PoseCandidates::MAX_CANDIDATES + blockSize.x - 1) / blockSize.x);

  ck_init_random_states<<<gridSize, blockSize>>>(randomStates, PoseCandidates::MAX_CANDIDATES, m_rngSeed);
}

void GPURansac_CUDA::generate_pose_candidates()
{
  const Vector2i imgSize = m_featureImage->noDims;
  const RGBDPatchFeature *features = m_featureImage->GetData(MEMORYDEVICE_CUDA);
  const GPUForestPrediction *predictions = m_predictionsImage->GetData(
      MEMORYDEVICE_CUDA);

  RandomState *randomStates = m_randomStates->GetData(MEMORYDEVICE_CUDA);
  PoseCandidates *poseCandidates = m_poseCandidates->GetData(MEMORYDEVICE_CUDA);

  dim3 blockSize(128);
  dim3 gridSize(
      (PoseCandidates::MAX_CANDIDATES + blockSize.x - 1) / blockSize.x);

  ck_generate_pose_candidates<<<gridSize, blockSize>>>(features, predictions, imgSize, randomStates,
      poseCandidates, PoseCandidates::MAX_CANDIDATES,
      m_useAllModesPerLeafInPoseHypothesisGeneration,
      m_checkMinDistanceBetweenSampledModes, m_minDistanceBetweenSampledModes,
      m_checkRigidTransformationConstraint,
      m_translationErrorMaxForCorrectPose);

  // Need to make the data available to the host
  m_poseCandidates->UpdateHostFromDevice();
}

void GPURansac_CUDA::compute_and_sort_energies()
{
  // Need to make the data available to the device
  m_poseCandidates->UpdateDeviceFromHost();

  const RGBDPatchFeature *features = m_featureImage->GetData(MEMORYDEVICE_CUDA);
  const GPUForestPrediction *predictions = m_predictionsImage->GetData(
      MEMORYDEVICE_CUDA);
  PoseCandidates *poseCandidates = m_poseCandidates->GetData(MEMORYDEVICE_CUDA);

  dim3 blockSize(128); // threads to compute the energy for each candidate
  dim3 gridSize(PoseCandidates::MAX_CANDIDATES); // Launch one block per candidate (many blocks will exit immediately in the later stages of ransac)
  ck_compute_energies<<<gridSize, blockSize>>>(features, predictions, poseCandidates);

  // Sort by ascending energy (start a single thread, the kernel will dispatch more threads as needed)
  ck_sort_energies<<<1,1>>>(poseCandidates);

  // Need to make the data available to the host once again
  m_poseCandidates->UpdateHostFromDevice();
}

}
