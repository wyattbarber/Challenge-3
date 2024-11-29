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

option(NOPYTHON "Build C library instead of Python package" OFF)


project(neuralnet LANGUAGES CXX)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(OpenMP REQUIRED)
find_package(Python3 3.11 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)


execute_process(
       COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/getversion.py "${CMAKE_SOURCE_DIR}"
       OUTPUT_VARIABLE TOML_PACKAGE_VERSION
   )
string(STRIP "${TOML_PACKAGE_VERSION}" TOML_PACKAGE_VERSION)
message(STATUS "Building neuralnet version ${TOML_PACKAGE_VERSION}")
set(PROJECT_VERSION ${TOML_PACKAGE_VERSION})


if(NOPYTHON)
    add_library(neuralnet STATIC)
    target_include_directories(neuralnet PRIVATE include)
    target_sources(neuralnet PRIVATE src/Training.cpp)
    target_link_libraries(neuralnet PRIVATE Eigen3::Eigen)
    
    include(CMakePackageConfigHelpers)
    include(GNUInstallDirs)

    install(
        DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/src
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    )

    install(
        TARGETS neuralnet
        EXPORT neuralnet-export
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    )

    configure_package_config_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/cmake/config.cmake.in
        neuralnet-config.cmake
        INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/neuralnet
    )

    write_basic_package_version_file(
        neuralnet-config-version.cmake
        VERSION ${PROJECT_VERSION}
        COMPATIBILITY AnyNewerVersion
    )

    export(
        EXPORT neuralnet-export
        NAMESPACE neuralnet::
        FILE neuralnet-targets.cmake
    )

    install(
        EXPORT neuralnet-export
        NAMESPACE neuralnet::
        FILE neuralnet-targets.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/neuralnet
    )

    install(
        FILES
            ${CMAKE_CURRENT_BINARY_DIR}/neuralnet-config.cmake
            ${CMAKE_CURRENT_BINARY_DIR}/neuralnet-config-version.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/neuralnet
    )

else()
    configure_file (version.h.in ${CMAKE_CURRENT_SOURCE_DIR}/version.h @ONLY) 
    
    pybind11_add_module(neuralnet bindings.cpp)
    target_link_libraries(neuralnet PRIVATE Eigen3::Eigen OpenMP::OpenMP_CXX)
    
    message(STATUS "Found Python3 installation location: ${Python3_SITEARCH}")
    install(TARGETS neuralnet
        COMPONENT python
        LIBRARY DESTINATION "${Python3_SITEARCH}"
        ARCHIVE DESTINATION "lib"
        RUNTIME DESTINATION "bin")
endif()