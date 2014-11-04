##########################
# SetCUDAAppTarget.cmake #
##########################

IF("${CMAKE_SYSTEM}" MATCHES "Linux")
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
ENDIF()

SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin/apps/${targetname})
SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${PROJECT_BINARY_DIR}/bin/apps/${targetname})
SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${PROJECT_BINARY_DIR}/bin/apps/${targetname})

IF(WITH_CUDA)
  CUDA_ADD_EXECUTABLE(${targetname} ${sources} ${headers} ${templates} OPTIONS --generate-code arch=compute_20,code=sm_20 --generate-code arch=compute_30,code=sm_30)
ELSE()
  ADD_EXECUTABLE(${targetname} ${sources} ${headers} ${templates})
ENDIF()

INCLUDE(${PROJECT_SOURCE_DIR}/cmake/VCLibraryHack.cmake)

IF(MSVC_IDE)
  SET_TARGET_PROPERTIES(${targetname} PROPERTIES LINK_FLAGS_DEBUG "/DEBUG")
ENDIF()
