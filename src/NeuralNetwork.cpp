#include <pybind11/pybind11.h>
namespace py = pybind11;


std::string helloworld(){
    return "Hello World!";
}

PYBIND11_MODULE(neuralnet, m){
    m.doc() ="CIS 678 Challenge #3 C++ backend";

    m.def("helloworld", &helloworld, "Build test function");
}