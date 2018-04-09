/**
 * itmx: MemoryBlockFactory.h
 * Copyright (c) Torr Vision Group, University of Oxford, 2015. All rights reserved.
 */

#ifndef H_ITMX_MEMORYBLOCKFACTORY
#define H_ITMX_MEMORYBLOCKFACTORY

#include <boost/shared_ptr.hpp>

#include <ORUtils/DeviceType.h>
#include <ORUtils/Image.h>
#include <ORUtils/MemoryBlock.h>

namespace itmx {

/**
 * \brief An instance of this class can be used to make memory blocks.
 */
class MemoryBlockFactory
{
  //#################### PRIVATE VARIABLES ####################
private:
  /** The type of device on which the memory blocks will primarily be used. */
  DeviceType m_deviceType;

  //#################### SINGLETON IMPLEMENTATION ####################
private:
  /**
   * \brief Constructs a memory block factory.
   */
  MemoryBlockFactory();

public:
  /**
   * \brief Gets the singleton instance.
   *
   * \return  The singleton instance.
   */
  static MemoryBlockFactory& instance();

  //#################### PUBLIC MEMBER FUNCTIONS ####################
public:
  /**
   * \brief Makes a memory block of the specified type and size.
   *
   * \param dataSize  The size of the memory block to make.
   * \return          The memory block.
   */
  template <typename T>
  boost::shared_ptr<ORUtils::MemoryBlock<T> > make_block(size_t dataSize = 0) const
  {
    bool allocateGPU = m_deviceType == DEVICE_CUDA;
    return boost::shared_ptr<ORUtils::MemoryBlock<T> >(new ORUtils::MemoryBlock<T>(dataSize, true, allocateGPU));
  }

  /**
   * \brief Makes an image of the specified type and size.
   *
   * \param size  The size of the image to make.
   * \return      The image.
   */
  template <typename T>
  boost::shared_ptr<ORUtils::Image<T> > make_image(const ORUtils::Vector2<int> size = ORUtils::Vector2<int>(0, 0)) const
  {
    bool allocateGPU = m_deviceType == DEVICE_CUDA;
    return boost::shared_ptr<ORUtils::Image<T> >(new ORUtils::Image<T>(size, true, allocateGPU));
  }

  /**
   * \brief Sets the type of device on which the memory blocks made by the factory will primarily be used.
   *
   * - If the device type is CPU, the memory blocks will only be allocated on the CPU.
   * - If the device type is CUDA, the memory blocks will be allocated on both the CPU and GPU.
   *
   * \param deviceType  The type of device on which the memory blocks made by the factory will primarily be used.
   */
  void set_device_type(DeviceType deviceType);
};

}

#endif
