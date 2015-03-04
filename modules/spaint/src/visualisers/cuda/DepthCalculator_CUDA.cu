/**
 * spaint: DepthCalculator_CUDA.cu
 */

#include "visualisers/cuda/DepthCalculator_CUDA.h"

#include "visualisers/shared/DepthCalculator_Shared.h"

namespace spaint {

//#################### CUDA KERNELS #################### 
__global__ void ck_render_euclidean_distance(float *outRendering, const Vector3f& cameraPosition, const Vector4f *ptsRay, Vector2i imgSize, float voxelSize)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= imgSize.x || y >= imgSize.y) return;

  int locId = y * imgSize.x + x;
  Vector4f ptRay = ptsRay[locId];
  shade_pixel_euclidean_distance(outRendering[locId], cameraPosition, ptRay.toVector3(), voxelSize, ptRay.w > 0);
}

__global__ void ck_render_orthographic_distance(float *outRendering, const Vector3f& cameraPosition, const Vector3f& cameraLookVector, const Vector4f *ptsRay, Vector2i imgSize, float voxelSize)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= imgSize.x || y >= imgSize.y) return;

  int locId = y * imgSize.x + x;
  Vector4f ptRay = ptsRay[locId];
  shade_pixel_orthographic_distance(outRendering[locId], cameraPosition, cameraLookVector, ptRay.toVector3(), voxelSize, ptRay.w > 0);
}

//#################### PUBLIC MEMBER FUNCTIONS #################### 

void DepthCalculator_CUDA::render_euclidean_distance(ITMFloatImage *outputImage, const ITMLib::Objects::ITMRenderState *renderState, const rigging::SimpleCamera *camera, float voxelSize) const
{
  // Shade all the pixels in the image.
  Vector2i imgSize = outputImage->noDims;
  const Eigen::Vector3f& cameraPositionEigen = camera->p();
  Vector3f cameraPosition;
  cameraPosition.x = cameraPositionEigen[0];
  cameraPosition.y = cameraPositionEigen[1];
  cameraPosition.z = cameraPositionEigen[2];

  dim3 cudaBlockSize(8, 8);
  dim3 gridSize((int)ceil((float)imgSize.x / (float)cudaBlockSize.x), (int)ceil((float)imgSize.y / (float)cudaBlockSize.y));
  ck_render_euclidean_distance<<<gridSize,cudaBlockSize>>>(
    outputImage->GetData(MEMORYDEVICE_CUDA),
    cameraPosition,
    renderState->raycastResult->GetData(MEMORYDEVICE_CUDA),
    imgSize,
    voxelSize
  );
}

void DepthCalculator_CUDA::render_orthographic_distance(ITMFloatImage *outputImage, const ITMLib::Objects::ITMRenderState *renderState, const rigging::SimpleCamera *camera, float voxelSize) const
{
  // Shade all the pixels in the image.
  Vector2i imgSize = outputImage->noDims;
  const Eigen::Vector3f& cameraPositionEigen = camera->p();
  Vector3f cameraPosition;
  cameraPosition.x = cameraPositionEigen[0];
  cameraPosition.y = cameraPositionEigen[1];
  cameraPosition.z = cameraPositionEigen[2];

  const Eigen::Vector3f& cameraLookVectorEigen = camera->n();
  Vector3f cameraLookVector;
  cameraLookVector.x = cameraLookVectorEigen[0];
  cameraLookVector.y = cameraLookVectorEigen[1];
  cameraLookVector.z = cameraLookVectorEigen[2];

  dim3 cudaBlockSize(8, 8);
  dim3 gridSize((int)ceil((float)imgSize.x / (float)cudaBlockSize.x), (int)ceil((float)imgSize.y / (float)cudaBlockSize.y));
  ck_render_orthographic_distance<<<gridSize,cudaBlockSize>>>(
    outputImage->GetData(MEMORYDEVICE_CUDA),
    cameraPosition,
    cameraLookVector,
    renderState->raycastResult->GetData(MEMORYDEVICE_CUDA),
    imgSize,
    voxelSize
  );
}
}

