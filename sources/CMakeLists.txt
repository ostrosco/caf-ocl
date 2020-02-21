set(clCAF.Source fft.cpp)
set(clCAF.Headers clCAF.h)

set(clCAF.Files ${clCAF.Source} ${clCAF.Headers})

add_executable(clCAF ${clCAF.Files})

include_directories(${OpenCL_INCLUDE_DIRS} ${CLFFT_INCLUDE_DIRS})
target_link_libraries(clCAF ${OpenCL_LIBRARIES} ${CLFFT_LIBRARIES})
set_target_properties(clCAF PROPERTIES VERSION ${CLCAF_VERSION})
set_target_properties(clCAF PROPERTIES SOVERSION ${CLCAF_SOVERSION})
