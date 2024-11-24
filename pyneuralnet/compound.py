from .abstract import Model, Encoder
from typing import List, Any

class Sequence(Model):
    _layers : List[Any]

    def __init__(self, *layers):
        """ 
        Construct a model of sequenced layers.

        :param layers: All layers, from input to output.
        """
        Model.__init__(self)
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


class DeepEncoder(Encoder):
    _layers : List[Any]

    def __init__(self, *layers):
        """ 
        Construct an encoder out of nested encoder components.

        :param layers: All layers, from input to latent layer.
        """
        Encoder.__init__(self)
        self._layers = layers

    def forward(self, input):
        return self.decode(self.encode(input))
    
    def backward(self, error):
        return self.backward_encode(self.backward_decode(error))
    
    def encode(self, input):
        h = [input]
        for l in self._layers:
            h.append(l.encode(h[-1]))
        return h[-1]
    
    def decode(self, embed):
        h = [embed]
        for l in self._layers[::-1]:
            h.append(l.decode(h[-1]))
        return h[-1]
    
    def backward_encode(self, error):
        h = [error]
        for l in self._layers[::-1]:
            h.append(l.backward_encode(h[-1]))
        return h[-1]
    
    def backward_decode(self, error):
        h = [error]
        for l in self._layers:
            h.append(l.backward_decode(h[-1]))
        return h[-1]
    
    def update(self, rate):
        for l in self._layers:
            l.update(rate)