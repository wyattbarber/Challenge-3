cmake_minimum_required(VERSION 3.15...3.29)

set(CMAKE_CXX_STANDARD 20 CACHE STRING "C++ version selection")
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CXX_FLAGS "-g")

project(neuralnet LANGUAGES CXX)

find_package(pybind11 CONFIG REQUIRED)
find_package (Eigen3 3.3 REQUIRED NO_MODULE)

pybind11_add_module(neuralnet src/bindings.cpp)
target_link_libraries(neuralnet PRIVATE Eigen3::Eigen)
install(TARGETS neuralnet DESTINATION .)