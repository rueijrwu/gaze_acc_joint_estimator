cmake_minimum_required(VERSION 3.24)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 60 CACHE STRING "CUDA device architectures")
endif()
project(acc_inverse VERSION 1.0.0
  DESCRIPTION "FP64 inverse optical calibration library for CPU and CUDA"
  LANGUAGES C CXX CUDA)
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
find_package(CUDAToolkit 12.9 REQUIRED)
find_package(Eigen3 3.4 REQUIRED NO_MODULE)
option(ACC_BUILD_CLI "Build the optional standalone HDF5 runtime" ON)
option(ACC_BUILD_TESTS "Build native C++ unit tests" ON)
if(ACC_BUILD_CLI)
  find_package(HDF5 REQUIRED COMPONENTS C CXX)
endif()
option(ACC_CUDA_FMAD "Allow FP64 CUDA fused multiply-add (no fast math)" ON)
option(ACC_CUDA_RESOURCE_REPORT "Print CUDA register/spill report" OFF)
string(REPLACE ";" "," ACC_ARCH_STRING "${CMAKE_CUDA_ARCHITECTURES}")
set(ACC_FLAG_STRING "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_RELEASE};CUDA_FMAD=${ACC_CUDA_FMAD};FP64;lineinfo")
configure_file(cmake/build_info.hpp.in generated/build_info.hpp @ONLY)
add_library(acc_inverse SHARED src/model_cpu.cpp src/model_cuda.cu src/solver.cpp src/api.cpp)
add_library(acc::inverse ALIAS acc_inverse)
target_compile_features(acc_inverse PUBLIC cxx_std_17)
target_include_directories(acc_inverse PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/generated")
target_link_libraries(acc_inverse PRIVATE Eigen3::Eigen CUDA::cudart CUDA::cublas CUDA::cusolver)
target_compile_options(acc_inverse PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
target_compile_definitions(acc_inverse PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:EIGEN_NO_CUDA>)
if(NOT ACC_CUDA_FMAD)
  target_compile_options(acc_inverse PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>)
endif()
if(ACC_CUDA_RESOURCE_REPORT)
  target_compile_options(acc_inverse PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-v>)
endif()
set_target_properties(acc_inverse PROPERTIES POSITION_INDEPENDENT_CODE ON
  VERSION ${PROJECT_VERSION} SOVERSION 1 EXPORT_NAME inverse
  INSTALL_RPATH_USE_LINK_PATH TRUE)
if(ACC_BUILD_CLI)
  add_executable(acc_inverse_cli apps/acc_inverse_cli.cpp)
  target_include_directories(acc_inverse_cli PRIVATE ${HDF5_INCLUDE_DIRS})
  target_link_libraries(acc_inverse_cli PRIVATE acc_inverse ${HDF5_CXX_LIBRARIES} ${HDF5_LIBRARIES})
  set_target_properties(acc_inverse_cli PROPERTIES
    INSTALL_RPATH "$ORIGIN/../${CMAKE_INSTALL_LIBDIR}" INSTALL_RPATH_USE_LINK_PATH TRUE)
endif()
if(ACC_BUILD_TESTS)
  enable_testing()
  add_executable(acc_inverse_tests tests/test_native.cpp)
  target_include_directories(acc_inverse_tests PRIVATE src)
  target_link_libraries(acc_inverse_tests PRIVATE acc_inverse Eigen3::Eigen CUDA::cudart)
  add_test(NAME native_cpu_math COMMAND acc_inverse_tests --cpu)
  add_test(NAME native_cuda_math COMMAND acc_inverse_tests --cuda)
  set_tests_properties(native_cuda_math PROPERTIES SKIP_RETURN_CODE 77)
  if(ACC_BUILD_CLI)
    add_executable(acc_inverse_hdf5_tests tests/test_hdf5.cpp)
    target_include_directories(acc_inverse_hdf5_tests PRIVATE include ${HDF5_INCLUDE_DIRS})
    target_link_libraries(acc_inverse_hdf5_tests PRIVATE ${HDF5_CXX_LIBRARIES} ${HDF5_LIBRARIES})
    add_test(NAME native_hdf5_roundtrip COMMAND acc_inverse_hdf5_tests $<TARGET_FILE:acc_inverse_cli>)
  endif()
endif()

configure_package_config_file(cmake/acc_inverseConfig.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/acc_inverseConfig.cmake"
  INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/acc_inverse")
write_basic_package_version_file("${CMAKE_CURRENT_BINARY_DIR}/acc_inverseConfigVersion.cmake"
  VERSION ${PROJECT_VERSION} COMPATIBILITY SameMajorVersion)
install(TARGETS acc_inverse EXPORT acc_inverseTargets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
if(TARGET acc_inverse_cli)
  install(TARGETS acc_inverse_cli RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(EXPORT acc_inverseTargets NAMESPACE acc::
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/acc_inverse")
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/acc_inverseConfig.cmake"
              "${CMAKE_CURRENT_BINARY_DIR}/acc_inverseConfigVersion.cmake"
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/acc_inverse")
