/**
 * spaint: ExampleReservoirs.tpp
 * Copyright (c) Torr Vision Group, University of Oxford, 2016. All rights reserved.
 */

#ifndef H_SPAINT_EXAMPLERESERVOIRSTPP
#define H_SPAINT_EXAMPLERESERVOIRSTPP

#include "randomforest/interface/ExampleReservoirs.h"
#include "util/MemoryBlockFactory.h"

namespace spaint
{

template<typename ExampleType, typename FeatureType, typename LeafType>
ExampleReservoirs<ExampleType, FeatureType, LeafType>::ExampleReservoirs(
    size_t capacity, size_t nbLeaves, uint32_t rngSeed)
{
  MemoryBlockFactory &mbf = MemoryBlockFactory::instance();

  m_reservoirCapacity = capacity;
  m_rngSeed = rngSeed;

  // One row per leaf
  m_data = mbf.make_image<ExampleType>(Vector2i(capacity, nbLeaves));
  m_reservoirsSize = mbf.make_block<int>(nbLeaves);
  m_reservoirsAddCalls = mbf.make_block<int>(nbLeaves);

  m_reservoirsSize->Clear();
  m_reservoirsAddCalls->Clear();
}

template<typename ExampleType, typename FeatureType, typename LeafType>
ExampleReservoirs<ExampleType, FeatureType, LeafType>::~ExampleReservoirs()
{
}

template<typename ExampleType, typename FeatureType, typename LeafType>
typename ExampleReservoirs<ExampleType, FeatureType, LeafType>::ExampleReservoirsImage_CPtr ExampleReservoirs<
    ExampleType, FeatureType, LeafType>::get_reservoirs() const
{
  return m_data;
}

template<typename ExampleType, typename FeatureType, typename LeafType>
ITMIntMemoryBlock_CPtr ExampleReservoirs<ExampleType, FeatureType, LeafType>::get_reservoirs_size() const
{
  return m_reservoirsSize;
}

template<typename ExampleType, typename FeatureType, typename LeafType>
int ExampleReservoirs<ExampleType, FeatureType, LeafType>::get_reservoirs_count() const
{
  return m_data->noDims.height;
}

template<typename ExampleType, typename FeatureType, typename LeafType>
int ExampleReservoirs<ExampleType, FeatureType, LeafType>::get_capacity() const
{
  return m_data->noDims.width;
}
}

#endif
