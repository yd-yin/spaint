/**
 * spaint: DepthVisualiser_CUDA.cu
 */

#include "visualisers/cuda/DepthVisualiser_CUDA.h"

#include "visualisers/shared/DepthVisualiser_Shared.h"

namespace spaint {

//#################### CUDA KERNELS ####################

__global__ void ck_render_depth(float *outRendering, const Vector4f *ptsRay, Vector3f cameraPosition, Vector3f cameraLookVector,
                                Vector2i imgSize, float voxelSize, float invalidDepthValue, DepthVisualiser::DepthType depthType)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
  if(x >= imgSize.x || y >= imgSize.y) return;

  int locId = y * imgSize.x + x;
  Vector4f ptRay = ptsRay[locId];
  shade_pixel_depth(outRendering[locId], ptRay.toVector3() * voxelSize, ptRay.w > 0, cameraPosition, cameraLookVector, voxelSize, invalidDepthValue, depthType);
}

//#################### PUBLIC MEMBER FUNCTIONS ####################

void DepthVisualiser_CUDA::render_depth(DepthType depthType, const Vector3f& cameraPosition, const Vector3f& cameraLookVector, const ITMLib::Objects::ITMRenderState *renderState,
                                        float voxelSize, float invalidDepthValue, const ITMFloatImage_Ptr& outputImage) const
{
  Vector2i imgSize = outputImage->noDims;

  // Shade all of the pixels in the image.
  dim3 cudaBlockSize(8, 8);
  dim3 gridSize((int)ceil((float)imgSize.x / (float)cudaBlockSize.x), (int)ceil((float)imgSize.y / (float)cudaBlockSize.y));

  ck_render_depth<<<gridSize,cudaBlockSize>>>(
    outputImage->GetData(MEMORYDEVICE_CUDA),
    renderState->raycastResult->GetData(MEMORYDEVICE_CUDA),
    cameraPosition,
    cameraLookVector,
    imgSize,
    voxelSize,
    invalidDepthValue,
    depthType
  );
}

}
