import importlib.metadata
from neuralnet import DataSource, DataSource2D
from . import loss
from . import training
from . import convolution
from . import fullconnect
from . import autoencoder

__version__ = importlib.metadata.version("pyneuralnet")

__backend_version__ = importlib.metadata.version("neuralnet")

if(__version__ != __backend_version__):
    raise ImportError(f"Imported backend version is out of sync with frontend.\
                      Imported {__backend_version__}, desired {__version__}.")
