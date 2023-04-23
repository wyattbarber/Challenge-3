import pytest
import numpy as np
import neuralnet as nn


@pytest.fixture(scope='module')
def data1():
    yield np.random.randn(100, 1000)


@pytest.fixture(scope='module')
def data2():
    yield np.random.randn(50, 1000)


def test_train(data1, data2):
    translator = nn.CoupledAutoencoder(
        [data1.shape[0]], 
        [data2.shape[0]], 
        40)
    errors = translator.train(data1, data2, 0.00001, 10, 0.99)