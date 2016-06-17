####################
# UseLodePNG.cmake #
####################

FIND_PATH(LodePNG_INCLUDE_DIR lodepng.h HINTS "${PROJECT_SOURCE_DIR}/libraries/lodepng-20160501/include")
INCLUDE_DIRECTORIES(${LodePNG_INCLUDE_DIR})

IF(MSVC_IDE)
  FIND_LIBRARY(LodePNG_LIBRARY_DEBUG lodepng_d HINTS "${PROJECT_SOURCE_DIR}/libraries/lodepng-20160501/build/lib")
  FIND_LIBRARY(LodePNG_LIBRARY_RELEASE lodepng HINTS "${PROJECT_SOURCE_DIR}/libraries/lodepng-20160501/build/lib")
  SET(LodePNG_LIBRARY debug ${LodePNG_LIBRARY_DEBUG} optimized ${LodePNG_LIBRARY_RELEASE})
ELSE()
  FIND_LIBRARY(LodePNG_LIBRARY lodepng HINTS "${PROJECT_SOURCE_DIR}/libraries/lodepng-20160501/build/lib")
ENDIF()
