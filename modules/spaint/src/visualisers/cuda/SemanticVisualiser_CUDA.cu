/**
 * spaint: SemanticVisualiser_CUDA.cu
 */

#include "visualisers/cuda/SemanticVisualiser_CUDA.h"

#include "visualisers/shared/SemanticVisualiser_Shared.h"

namespace spaint {

//#################### CUDA KERNELS ####################

__global__ void ck_render_semantic(Vector4u *outRendering, const Vector4f *ptsRay, const SpaintVoxel *voxelData, const ITMVoxelIndex::IndexData *voxelIndex,
                                   Vector2i imgSize, Vector3u *labelColours, Vector3f viewerPos, Vector3f lightPos)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= imgSize.x || y >= imgSize.y) return;

  int locId = y * imgSize.x + x;
  Vector4f ptRay = ptsRay[locId];
  shade_pixel_semantic(outRendering[locId], ptRay.toVector3(), ptRay.w > 0, voxelData, voxelIndex, labelColours, viewerPos, lightPos);
}

//#################### PUBLIC MEMBER FUNCTIONS ####################

void SemanticVisualiser_CUDA::render(const ITMLib::Objects::ITMScene<SpaintVoxel,ITMVoxelIndex> *scene, const ITMLib::Objects::ITMPose *pose,
                                     const ITMLib::Objects::ITMIntrinsics *intrinsics, const ITMLib::Objects::ITMRenderState *renderState,
                                     ITMUChar4Image *outputImage) const
{
  // Set up the label colours.
  // FIXME: These should ultimately be passed in from elsewhere.
  ORUtils::MemoryBlock<Vector3u> labelColours(4 * sizeof(Vector3u), true, true);
  Vector3u *labelColoursData = labelColours.GetData(MEMORYDEVICE_CPU);
  labelColoursData[0] = Vector3u(255, 255, 255);
  labelColoursData[1] = Vector3u(255, 0, 0);
  labelColoursData[2] = Vector3u(0, 255, 0);
  labelColoursData[3] = Vector3u(0, 0, 255);
  labelColours.UpdateDeviceFromHost();

  // Shade all of the pixels in the image.
  Vector2i imgSize = outputImage->noDims;
  Vector3f lightPos(0.0f, -100.0f, 0.0f);
  Vector3f viewerPos(pose->GetInvM().getColumn(3));

  dim3 cudaBlockSize(8, 8);
  dim3 gridSize((int)ceil((float)imgSize.x / (float)cudaBlockSize.x), (int)ceil((float)imgSize.y / (float)cudaBlockSize.y));

  ck_render_semantic<<<gridSize,cudaBlockSize>>>(
    outputImage->GetData(MEMORYDEVICE_CUDA),
    renderState->raycastResult->GetData(MEMORYDEVICE_CUDA),
    scene->localVBA.GetVoxelBlocks(),
    scene->index.getIndexData(),
    imgSize,
    labelColours.GetData(MEMORYDEVICE_CUDA),
    viewerPos,
    lightPos
  );
}

}
