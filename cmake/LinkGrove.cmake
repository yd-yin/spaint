###################
# LinkGrove.cmake #
###################

IF(BUILD_GROVE)
  TARGET_LINK_LIBRARIES(${targetname} grove tvgutil itmx ${CUDA_cudadevrt_LIBRARY})

  INCLUDE(${PROJECT_SOURCE_DIR}/cmake/LinkScoreForests.cmake)
ENDIF()
