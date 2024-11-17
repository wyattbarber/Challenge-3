SCRIPT_PATH=$(dirname $(realpath -s $0))
# Compile module manually with debugging settings
c++ -O3 -fopenmp -Wall -shared -g -std=c++20 -I ~/include/eigen/ -I ~/include/eigen/unsupported/ -fPIC $(python3 -m pybind11 --includes) $SCRIPT_PATH/../bindings.cpp -o $SCRIPT_PATH/neuralnet$(python3-config --extension-suffix)
# Run any debugging steps on the test script
valgrind --leak-check=yes --track-origins=yes --suppressions="$SCRIPT_PATH/python-valgrind.supp" --log-file="$SCRIPT_PATH/valgrind.txt" python3 -E -tt -m debug.debug $SCRIPT_PATH > "$SCRIPT_PATH/pylog.txt"