import pytest
import numpy as np
import neuralnet as nn

def test_construct():
    model = nn.Network([9,5,1], [ nn.ActivationFunctions.ReLU, nn.ActivationFunctions.ReLU])

# Simply test that method runs without exceptions and output is in reasonable range
def test_forward_pass():
    model = nn.Network([9, 5, 1], [ nn.ActivationFunctions.ReLU, nn.ActivationFunctions.ReLU])

    output = model.forward([1 if i%2 else 0 for i in range(10)])
    assert all([abs(o) <= 1 for o in output])

# Tests that backpropagation runs without error
def test_bwdpass():
    model = nn.Network([9, 5, 1], [ nn.ActivationFunctions.ReLU, nn.ActivationFunctions.ReLU])
    train = [
        ([0,0,0,0,0,0,0,0,0], [1]),
        ([1,0,0,1,0,0,1,0,0], [0]),
        ([0,1,0,1,0,1,0,0,0], [1]),
        ([0,1,0,1,1,0,0,0,0], [0]),
        ([0,0,0,0,0,0,1,1,0], [1]),
        ([1,1,1,1,1,1,1,1,1], [0]),
    ]
    model.train([d[0] for d in train], [d[1] for d in train], 1, 5)

# Test if nn implementation can learn to classify normally distributed data
@pytest.mark.parametrize("data,exp",[
        (np.random.normal(-1, 1, 10), 1),
        (np.random.normal(-1, 1, 10), 1),
        (np.random.normal(1, 1, 10), 0),
        (np.random.normal(1, 1, 10), 0)
    ]
)
def test_normal(data, exp):
    model = nn.Network([10, 2, 1], [nn.ActivationFunctions.ReLU, nn.ActivationFunctions.Sigmoid])

    train = []
    test = []
    for i in range(100):
        if i % 2 == 0:
            t = np.random.normal(-1, 1, 10)
            e = [1]
        else:
            t = np.random.normal(1, 1, 10)
            e = [0]
        train.append(t)
        test.append(e)
    
    model.train(train, test, 0.5, 10)

    if exp == 1:
        assert model.forward(data)[0] > 0.8
    else:
        assert model.forward(data)[0] < 0.2



# Test if nn implementation can learn to classify normally distributed data
@pytest.mark.parametrize("data,exp",[
        (np.random.normal(-1, 1, 10), [0,1]),
        (np.random.normal(-1, 1, 10), [0,1]),
        (np.random.normal(1, 1, 10), [1,0]),
        (np.random.normal(1, 1, 10), [1,0])
    ]
)
def test_softmax(data, exp):
    model = nn.Network([10, 2, 2], [nn.ActivationFunctions.ReLU, nn.ActivationFunctions.SoftMax])

    train = []
    test = []
    for i in range(100):
        if i % 2 == 0:
            t = np.random.normal(-1, 1, 10)
            e = [0,1]
        else:
            t = np.random.normal(1, 1, 10)
            e =  [1,0]
        train.append(t)
        test.append(e)
    
    model.train(train, test, 0.5, 10)

    if exp == [0,1]:
        assert model.forward(data)[1] > 0.8 and model.forward(data)[0] < 0.2
    else:
        assert model.forward(data)[0] > 0.8 and model.forward(data)[1] < 0.2

@pytest.mark.skip
def test_test():
    train = []
    test = []
    for i in range(100):
        if i % 2 == 0:
            t = np.random.normal(-1, 1, 10)
            e = [0,1]
        else:
            t = np.random.normal(1, 1, 10)
            e =  [1,0]
        train.append(t)
        test.append(e)
    
    error = nn.test([10, 2, 2], [nn.ActivationFunctions.ReLU, nn.ActivationFunctions.SoftMax], train, test, 0.05, 5, 0.5, 10)

    assert (np.mean(error) >= 0) and (np.mean(error) < 0.5), "Model error was nonsensical or didn't reflect proper training occuring"
