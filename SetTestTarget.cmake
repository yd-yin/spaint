#######################
# SetTestTarget.cmake #
#######################

SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin/tests/${targetname})
SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${PROJECT_BINARY_DIR}/bin/tests/${targetname})
SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${PROJECT_BINARY_DIR}/bin/tests/${targetname})
ADD_EXECUTABLE(${targetname} ${sources} ${headers} ${templates})
INCLUDE(${PROJECT_SOURCE_DIR}/VCLibraryHack.cmake)

