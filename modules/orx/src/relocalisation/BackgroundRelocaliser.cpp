/**
 * orx: BackgroundRelocaliser.cpp
 * Copyright (c) Torr Vision Group, University of Oxford, 2017. All rights reserved.
 */

#include "relocalisation/BackgroundRelocaliser.h"

#include <iostream>

namespace orx {

//#################### CONSTRUCTORS ####################

BackgroundRelocaliser::BackgroundRelocaliser(const Relocaliser_Ptr& relocaliser, int relocalisationDevice)
: m_relocalisationDevice(relocalisationDevice), m_relocaliser(relocaliser), m_relocaliserRunning(false)
{}

//#################### PUBLIC MEMBER FUNCTIONS ####################

void BackgroundRelocaliser::finish_training()
{
  // Set the current GPU to the one on which calls to the decorated relocaliser should be performed.
  to_relocalisation_gpu();

  // Signal to the decorated relocaliser that no more calls will be made to its train or update functions.
  m_relocaliser->finish_training();

  // Reset the current GPU to the one on which calls were previously being performed.
  to_old_gpu();
}

ORUChar4Image_CPtr BackgroundRelocaliser::get_visualisation_image(const std::string& key) const
{
  // Set the current GPU to the one on which calls to the decorated relocaliser should be performed.
  to_relocalisation_gpu();

  // Get the visualisation image from the decorated relocaliser.
  ORUChar4Image_CPtr result = m_relocaliser->get_visualisation_image(key);

  // Reset the current GPU to the one on which calls were previously being performed.
  to_old_gpu();

  return result;
}

void BackgroundRelocaliser::load_from_disk(const std::string& inputFolder)
{
  // Set the current GPU to the one on which calls to the decorated relocaliser should be performed.
  to_relocalisation_gpu();

  // Load the decorated relocaliser from disk.
  m_relocaliser->load_from_disk(inputFolder);

  // Reset the current GPU to the one on which calls were previously being performed.
  to_old_gpu();
}

std::vector<Relocaliser::Result> BackgroundRelocaliser::relocalise(const ORUChar4Image *colourImage, const ORFloatImage *depthImage, const Vector4f& depthIntrinsics) const
{
  // Prevent training and updating of the decorated relocaliser during a relocalisation.
  m_relocaliserRunning = true;

  // Copy the colour and depth images we want to use for relocalisation across to the CPU.
  colourImage->UpdateHostFromDevice();
  depthImage->UpdateHostFromDevice();

  // Set the current GPU to the one on which calls to the decorated relocaliser should be performed.
  to_relocalisation_gpu();

  // Make internal copies of the colour and depth images that are accessible on the new GPU.
  copy_images(colourImage, depthImage);

  // Attempt to relocalise using the internal copies.
  std::vector<Relocaliser::Result> results = m_relocaliser->relocalise(m_colourImage.get(), m_depthImage.get(), depthIntrinsics);

  // Reset the current GPU to the one on which calls were previously being performed.
  to_old_gpu();

  // Allow training and updating of the decorated relocaliser again.
  m_relocaliserRunning = false;

  return results;
}

void BackgroundRelocaliser::reset()
{
  // Set the current GPU to the one on which calls to the decorated relocaliser should be performed.
  to_relocalisation_gpu();

  // Reset the decorated relocaliser.
  m_relocaliser->reset();

  // Reset the current GPU to the one on which calls were previously being performed.
  to_old_gpu();
}

void BackgroundRelocaliser::save_to_disk(const std::string& outputFolder) const
{
  // Set the current GPU to the one on which calls to the decorated relocaliser should be performed.
  to_relocalisation_gpu();

  // Save the decorated relocaliser to disk.
  m_relocaliser->save_to_disk(outputFolder);

  // Reset the current GPU to the one on which calls were previously being performed.
  to_old_gpu();
}

void BackgroundRelocaliser::train(const ORUChar4Image *colourImage, const ORFloatImage *depthImage,
                                  const Vector4f& depthIntrinsics, const ORUtils::SE3Pose& cameraPose)
{
  // If a relocalisation is in progress, avoid trying to train the decorated relocaliser.
  if(m_relocaliserRunning) return;

  // Copy the colour and depth images we want to use for training across to the CPU.
  colourImage->UpdateHostFromDevice();
  depthImage->UpdateHostFromDevice();

  // Set the current GPU to the one on which calls to the decorated relocaliser should be performed.
  to_relocalisation_gpu();

  // Make internal copies of the colour and depth images that are accessible on the new GPU.
  copy_images(colourImage, depthImage);

  // Train the decorated relocaliser using the internal copies.
  m_relocaliser->train(m_colourImage.get(), m_depthImage.get(), depthIntrinsics, cameraPose);

  // Reset the current GPU to the one on which calls were previously being performed.
  to_old_gpu();
}

void BackgroundRelocaliser::update()
{
  // If a relocalisation is in progress, avoid trying to update the decorated relocaliser.
  if(m_relocaliserRunning) return;

  // Set the current GPU to the one on which calls to the decorated relocaliser should be performed.
  to_relocalisation_gpu();

  // Update the decorated relocaliser.
  m_relocaliser->update();

  // Reset the current GPU to the one on which calls were previously being performed.
  to_old_gpu();
}

//#################### PRIVATE MEMBER FUNCTIONS ####################

void BackgroundRelocaliser::copy_images(const ORUChar4Image *colourImage, const ORFloatImage *depthImage) const
{
  // If the internal images do not yet exist, create them.
  if(!m_colourImage) m_colourImage.reset(new ORUChar4Image(colourImage->noDims, true, true));
  if(!m_depthImage) m_depthImage.reset(new ORFloatImage(depthImage->noDims, true, true));

  // Make sure that the internal images have the same size as the input images we are trying to copy into them.
  m_colourImage->ChangeDims(colourImage->noDims);
  m_depthImage->ChangeDims(depthImage->noDims);

  // Copy the input images into the internal images on the CPU.
  m_colourImage->SetFrom(colourImage, ORUChar4Image::CPU_TO_CPU);
  m_depthImage->SetFrom(depthImage, ORFloatImage::CPU_TO_CPU);

  // Copy the contents of the internal images across to the GPU.
  m_colourImage->UpdateDeviceFromHost();
  m_depthImage->UpdateDeviceFromHost();
}

void BackgroundRelocaliser::to_old_gpu() const
{
  ORcudaSafeCall(cudaSetDevice(m_oldDevice));
}

void BackgroundRelocaliser::to_relocalisation_gpu() const
{
  ORcudaSafeCall(cudaGetDevice(&m_oldDevice));
  ORcudaSafeCall(cudaSetDevice(m_relocalisationDevice));
}

}
