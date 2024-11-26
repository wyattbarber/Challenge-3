import importlib.metadata
import neuralnet
from . import abstract
from . import loss
from . import training
from . import convolution
from . import fullconnect
from . import autoencoder
from . import compound
from . import normalize

__version__ = importlib.metadata.version("pyneuralnet")

if(__version__ != neuralnet.__version__):
    raise ImportError(f"Imported backend version is out of sync with frontend. Imported {neuralnet.__version__}, desired {__version__}.")
