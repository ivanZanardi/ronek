# Future imports
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Package
__author__ = "Ivan Zanardi"
__email__ = "zanardi3@illinois.edu"
__url__ = "https://github.com/ivanZanardi/ronek"
__license__ = "MIT License"
__version__ = "0.0.1"

__all__ = [
  "backend",
  "const",
  "env",
  "postproc",
  "roms",
  "systems"
]

# Ignore all warnings
import warnings
warnings.filterwarnings("ignore")

# Logging
from absl import logging
logging.set_verbosity(logging.ERROR)
