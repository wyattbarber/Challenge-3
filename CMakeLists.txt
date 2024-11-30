cmake_minimum_required(VERSION 3.15...3.29)

set(CMAKE_CXX_STANDARD 20 CACHE STRING "C++ version selection")
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

if(WIN32)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Wall /openmp:experimental")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /DEBUG /DPYBIND11_DETAILED_ERROR_MESSAGES")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2 /GL /DNDEBUG")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -DPYBIND11_DETAILED_ERROR_MESSAGES")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -DNDEBUG")
endif()


project(neuralnet LANGUAGES CXX)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(OpenMP REQUIRED)

execute_process(
       COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/getversion.py "${CMAKE_SOURCE_DIR}"
       OUTPUT_VARIABLE TOML_PACKAGE_VERSION
   )
string(STRIP "${TOML_PACKAGE_VERSION}" TOML_PACKAGE_VERSION)
message(STATUS "Building neuralnet version ${TOML_PACKAGE_VERSION}")
set(PROJECT_VERSION ${TOML_PACKAGE_VERSION})
configure_file (version.h.in ${CMAKE_CURRENT_SOURCE_DIR}/version.h @ONLY) 

# Workaround to installing C++ package to get EGR 532 project working
install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/ DESTINATION "C:/Users/wyatt/Documents/GVSU/EGR532/FinalProject/src/neuralnet")

# Install python module
find_package(Python3 3.11 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(neuralnet bindings.cpp)
target_link_libraries(neuralnet PRIVATE Eigen3::Eigen OpenMP::OpenMP_CXX)

message(STATUS "Found Python3 installation location: ${Python3_SITEARCH}")
install(TARGETS neuralnet
    COMPONENT python
    LIBRARY DESTINATION "${Python3_SITEARCH}"
    ARCHIVE DESTINATION "lib"
    RUNTIME DESTINATION "bin")