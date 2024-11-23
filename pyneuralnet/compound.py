from neuralnet import Model, Model2D, Encoder, Encoder2D, LossBase
from typing import List, Any

class Sequence(Model, Model2D):
    _layers : List[Any]

    def __init__(self, *layers):
        """ 
        Construct a model of sequenced layers.

        :param layers: All layers, from input to output.
        """
        Model.__init__(self)
        Model2D.__init__(self)
        self._layers = layers

    def forward(self, input):
        h = [input]
        for l in self._layers:
            h.append(l.forward(h[-1]))
        return h[-1]
    
    def backward(self, error):
        h = [error]
        for l in self._layers[::-1]:
            h.append(l.backward(h[-1]))
        return h[-1]
    
    def update(self, rate):
        for l in self._layers:
            l.update(rate)
