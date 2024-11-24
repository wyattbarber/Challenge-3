from neuralnet import DataSource, DataSource2D, LossBase
from neuralnet import Model as _Model
from neuralnet import Model2D as _Model2D
from neuralnet import Encoder as _Encoder
from neuralnet import Encoder2D as _Encoder2D


class Model(_Model, _Model2D):
    """
    Abstract base for model objects.

    Defines default implementations for common python
    operations, such as pickling.
    """
    def __init__(self):
        _Model.__init__(self)
        _Model2D.__init__(self)

    def __getstate__(self):
        return self.__dict__
    
    def __setstate__(self, state):
        self.__dict__  = state


class Encoder(Model, _Encoder, _Encoder2D):
    """
    Abstract base for encoder objects.

    Defines default implementations for common python
    operations, such as pickling.
    """
    def __init__(self):
        Model.__init__(self)
        _Encoder.__init__(self)
        _Encoder2D.__init__(self)
