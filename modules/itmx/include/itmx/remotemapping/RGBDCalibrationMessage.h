/**
 * itmx: RGBDCalibrationMessage.h
 * Copyright (c) Torr Vision Group, University of Oxford, 2017. All rights reserved.
 */

#ifndef H_ITMX_RGBDCALIBRATIONMESSAGE
#define H_ITMX_RGBDCALIBRATIONMESSAGE

#include <ITMLib/Objects/Camera/ITMRGBDCalib.h>

#include "DepthCompressionType.h"
#include "MappingMessage.h"
#include "RGBCompressionType.h"

namespace itmx {

/**
 * \brief An instance of this class represents a message containing the calibration for an RGB-D camera.
 */
class RGBDCalibrationMessage : public MappingMessage
{
  //#################### PRIVATE VARIABLES ####################
private:
  /** The byte segment within the message data that corresponds to the calibration parameters. */
  Segment m_calibSegment;

  /** The type of compression applied to the depth images. */
  Segment m_depthCompressionTypeSegment;

  /** The byte segment within the message data that corresponds to the depth image size. */
  Segment m_depthImageSizeSegment;

  /** The type of compression applied to the RGB images. */
  Segment m_rgbCompressionTypeSegment;

  /** The byte segment within the message data that corresponds to the RGB image size. */
  Segment m_rgbImageSizeSegment;

  //#################### CONSTRUCTORS ####################
public:
  /**
   * \brief Constructs an RGB-D calibration message.
   */
  RGBDCalibrationMessage();

  //#################### PUBLIC MEMBER FUNCTIONS ####################
public:
  /**
   * \brief Extracts the cameras's calibration parameters from the message.
   *
   * \return  The camera's calibration parameters.
   */
  ITMLib::ITMRGBDCalib extract_calib() const;

  /**
   * \brief Extracts the type of compression applied to the depth images.
   *
   * \return  The type of compression applied to the depth images.
   */
  DepthCompressionType extract_depth_compression_type() const;

  /**
   * \brief Extracts the camera's depth image size from the message.
   *
   * \return  The camera's depth image size.
   */
  Vector2i extract_depth_image_size() const;

  /**
   * \brief Extracts the type of compression applied to the RGB images.
   *
   * \return  The type of compression applied to the RGB images.
   */
  RGBCompressionType extract_rgb_compression_type() const;

  /**
   * \brief Extracts the camera's RGB image size from the message.
   *
   * \return  The camera's RGB image size.
   */
  Vector2i extract_rgb_image_size() const;

  /**
   * \brief Copies the camera's calibration parameters into the appropriate byte segment in the message.
   *
   * \param calib The camera's calibration parameters.
   */
  void set_calib(const ITMLib::ITMRGBDCalib& calib);

  /**
   * \brief Sets the type of compression applied to the depth images.
   *
   * \param depthCompressionType  The type of compression applied to the depth images.
   */
  void set_depth_compression_type(DepthCompressionType depthCompressionType);

  /**
   * \brief Copies the camera's depth image size into the appropriate byte segment in the message.
   *
   * \param depthImageSize  The camera's depth image size.
   */
  void set_depth_image_size(const Vector2i& depthImageSize);

  /**
   * \brief Sets the type of compression applied to the RGB images.
   *
   * \param rgbCompressionType  The type of compression applied to the RGB images.
   */
  void set_rgb_compression_type(RGBCompressionType rgbCompressionType);

  /**
   * \brief Copies the camera's RGB image size into the appropriate byte segment in the message.
   *
   * \param rgbImageSize  The camera's RGB image size.
   */
  void set_rgb_image_size(const Vector2i& rgbImageSize);
};

}

#endif
