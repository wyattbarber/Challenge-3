import pytest
import numpy as np
import neuralnet as nn

def test_construct():
    model = nn.Network([9,5,1], [ nn.ActivationFunctions.ReLU, nn.ActivationFunctions.ReLU])

# Simply test that method runs without exceptions and output is in reasonable range
def test_forward_pass():
    model = nn.Network([9, 5, 1], [ nn.ActivationFunctions.ReLU, nn.ActivationFunctions.Sigmoid])

    output = model.forward([1 if i%2 else 0 for i in range(10)])
    assert all([o <= 1 and o >= 0 for o in output])

# Tests that backpropagation runs without error
def test_bwdpass():
    model = nn.Network([9, 5, 2], [ nn.ActivationFunctions.ReLU, nn.ActivationFunctions.ReLU])
    train = [
        ([0,0,0,0,0,0,0,0,0], 1),
        ([1,0,0,1,0,0,1,0,0], 0),
        ([0,1,0,1,0,1,0,0,0], 1),
        ([0,1,0,1,1,0,0,0,0], 0),
        ([0,0,0,0,0,0,1,1,0], 1),
        ([1,1,1,1,1,1,1,1,1], 0),
    ]
    model.train([d[0] for d in train], [d[1] for d in train], 1, 5)

# Test if nn implementation can learn to classify normally distributed data
@pytest.mark.parametrize("data,exp",[
        (np.random.normal(-1, 1, 10), 0),
        (np.random.normal(-1, 1, 10), 0),
        (np.random.normal(1, 1, 10), 1),
        (np.random.normal(1, 1, 10), 1)
    ]
)
def test_softmax(data, exp):
    model = nn.Network([10, 2, 2], [nn.ActivationFunctions.ReLU, nn.ActivationFunctions.SoftMax])

    train = []
    test = []
    for i in range(100):
        if i % 2 == 0:
            t = np.random.normal(-1, 1, 10)
            e = 0
        else:
            t = np.random.normal(1, 1, 10)
            e = 1
        train.append(t)
        test.append(e)
    
    model.train(train, test, 0.5, 10)

    if exp == 1:
        assert model.forward(data)[1] > 0.8 and model.forward(data)[0] < 0.2
    else:
        assert model.forward(data)[0] > 0.8 and model.forward(data)[1] < 0.2


def test_test():
    train = []
    test = []
    for i in range(100):
        if i % 2 == 0:
            t = np.random.normal(-1, 1, 10)
            e = 0
        else:
            t = np.random.normal(1, 1, 10)
            e = 1
        train.append(t)
        test.append(e)
    
    error = nn.test([10, 2, 2], [nn.ActivationFunctions.ReLU, nn.ActivationFunctions.SoftMax], train, test, 0.05, 5, 0.5, 10)

    assert (np.mean(error) >= 0) and (np.mean(error) < 0.3), "Model error was nonsensical or didn't reflect proper training occuring"


# Test if adam network runs and converges same as unoptimized network
@pytest.mark.parametrize("data,exp",[
        (np.random.normal(-1, 1, 10), 0),
        (np.random.normal(-1, 1, 10), 0),
        (np.random.normal(1, 1, 10), 1),
        (np.random.normal(1, 1, 10), 1)
    ]
)
def test_adam(data, exp):
    model = nn.AdamNetwork([10, 2, 2], [nn.ActivationFunctions.ReLU, nn.ActivationFunctions.SoftMax])

    train = []
    test = []
    for i in range(100):
        if i % 2 == 0:
            t = np.random.normal(-1, 1, 10)
            e = 0
        else:
            t = np.random.normal(1, 1, 10)
            e = 1
        train.append(t)
        test.append(e)
    
    model.train(train, test, 0.5, 10, 0.9, 0.9)

    if exp == 1:
        assert model.forward(data)[1] > 0.8 and model.forward(data)[0] < 0.2
    else:
        assert model.forward(data)[0] > 0.8 and model.forward(data)[1] < 0.2


def test_test_dims():
    train = []
    test = []
    for i in range(100):
        if i % 2 == 0:
            t = np.random.normal(-1, 1, 10)
            e = 0
        else:
            t = np.random.normal(1, 1, 10)
            e = 1
        train.append(t)
        test.append(e)
    
    errors = nn.test_layers(
        [[10, 5, 2], [10, 2, 2], [10, 7, 2]],
        [[nn.ActivationFunctions.ReLU, nn.ActivationFunctions.SoftMax],[nn.ActivationFunctions.ReLU, nn.ActivationFunctions.SoftMax],[nn.ActivationFunctions.ReLU, nn.ActivationFunctions.SoftMax]],
        train, test, 0.05, 5, 0.5, 10
    )

    for error in errors:
        assert (np.mean(error) >= 0) and (np.mean(error) < 0.3), "Model error was nonsensical or didn't reflect proper training occuring"