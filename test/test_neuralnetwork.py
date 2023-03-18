import pytest
import neuralnet as nn

def test_construct():
    model = nn.NeuralNetwork([9,5,1])

# Simply test that method runs without exceptions and output is in reasonable range
def test_forward_pass():
    model = nn.NeuralNetwork([9, 5, 1])
    output = model.forwardPass([1 if i%2 else 0 for i in range(10)])
    assert all([abs(o) <= 1 for o in output])
    assert not all([o == 0 for o in output])

# Tests that backpropagation runs without error
def test_bwdpass():
    model = nn.NeuralNetwork([9, 5, 1])
    train = [
        ([0,0,0,0,0,0,0,0,0], [1]),
        ([1,0,0,1,0,0,1,0,0], [0]),
        ([0,1,0,1,0,1,0,0,0], [1]),
        ([0,1,0,1,1,0,0,0,0], [0]),
        ([0,0,0,0,0,0,1,1,0], [1]),
        ([1,1,1,1,1,1,1,1,1], [0]),
    ]
    model.backprop([d[0] for d in train], [d[1] for d in train], 0.1, 5)

# Test if nn implementation can learn XOR logic
@pytest.mark.parametrize("data,exp",[
        ([0,0], [0]),
        ([1,0], [1]),
        ([0,1], [1]),
        ([1,1], [0]),
    ]
)
def test_xor(data, exp):
    model = nn.NeuralNetwork([2, 5, 1])
    train = [
        ([0,0], [0]),
        ([1,0], [1]),
        ([0,1], [1]),
        ([1,1], [0]),
    ]
    model.backprop([d[0] for d in train], [d[1] for d in train], 0.1, 5)

    if exp == 1:
        assert model.forwardPass(data)[0] > 0.9
    else:
        assert model.forwardPass(data)[0] < 0.1

