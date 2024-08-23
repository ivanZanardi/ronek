"""
Build balanced truncation-based ROM.
"""

import sys
import json
import argparse
import importlib

# Inputs
# =====================================
parser = argparse.ArgumentParser()
parser.add_argument("--inpfile", type=str, help="path to JSON input file")
args = parser.parse_args()

with open(args.inpfile) as file:
  inputs = json.load(file)

# Import 'ronek' package
# =====================================
if (importlib.util.find_spec("ronek") is None):
  sys.path.append(inputs["path_to_lib"])

# Environment
# =====================================
from ronek import env
env.set(**inputs["env"])

# Libraries
# =====================================
import numpy as np

from ronek import utils
from ronek import systems as sys_mod
from ronek.roms import BalancedTruncation

# Main
# =====================================
if (__name__ == '__main__'):

  # Isothermal master equation model
  # -----------------------------------
  path_to_dtb = inputs["paths"]["dtb"]
  system = utils.get_class(
    modules=[sys_mod],
    name=inputs["system"]["name"]
  )(
    rates=path_to_dtb + "/kinetics.hdf5",
    species={
      k: path_to_dtb + f"/species/{k}.json" for k in ("atom", "molecule")
    },
    **inputs["system"]["kwargs"]
  )

  # Balanced truncation
  # -----------------------------------
  # Time and internal temperature grids
  t = system.get_tgrid(**inputs["grids"]["t"])
  Tint = np.geomspace(**inputs["grids"]["Tint"])
  # Model reduction
  max_mom = int(inputs["max_mom"])
  btrunc = BalancedTruncation(
    operators=system.compute_lin_fom_ops(Tint=Tint, max_mom=max_mom),
    path_to_saving=inputs["paths"]["saving"] + f"/max_mom_{max_mom}/",
    **inputs["btrunc"]
  )
  btrunc(t)
