# Specify the CUDA architectures to target
set(CMAKE_CUDA_ARCHITECTURES 60 70 75 80)

###############################################################################
##                                                                            
## Binary creation (All link_directories() calls must be made before this point.)
##
###############################################################################

# Collect the main source file
file(GLOB_RECURSE STORM_CUDA_MAIN_FILE ${PROJECT_SOURCE_DIR}/src/storm-cuda/storm_cuda.cpp)

# Collect other source files
file(GLOB_RECURSE STORM_CUDA_SOURCE_FILES ${PROJECT_SOURCE_DIR}/src/storm-cuda/StormcuSpar.cpp)

# Include additional directories
include_directories(${ADDITIONAL_INCLUDE_DIRS})

# Find the cuSPARSE library
find_package(CUDA REQUIRED)
find_library(CUSPARSE_LIBRARY cusparse HINTS ${CUDA_TOOLKIT_ROOT_DIR}/lib64)

# Find Boost libraries
find_package(Boost REQUIRED COMPONENTS system filesystem thread) # Added thread component
include_directories(${Boost_INCLUDE_DIRS})

# Create storm executable
add_executable(storm_cuda_executable ${STORM_CUDA_MAIN_FILE} ${STORM_CUDA_SOURCE_FILES})
target_link_libraries(storm_cuda_executable PUBLIC storm storm-cli-utilities ${CUDA_LIBRARIES} ${CUSPARSE_LIBRARY} ${Boost_LIBRARIES})
set_target_properties(storm_cuda_executable PROPERTIES OUTPUT_NAME "storm_cuda")
target_precompile_headers(storm_cuda_executable PRIVATE ${STORM_PRECOMPILED_HEADERS})

###############################################################################
##                                                                            
##  
##
###############################################################################
