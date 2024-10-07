"""
Build balanced truncation-based ROM.
"""

import os
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

from tqdm import tqdm
from ronek import utils
from ronek import systems as sys_mod
from ronek.roms import BalancedTruncation

# Main
# =====================================
if (__name__ == '__main__'):

  # Isothermal master equation system
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
  # Initialization
  # ---------------
  # Path to saving
  path_to_saving = inputs["paths"]["saving"]
  os.makedirs(path_to_saving, exist_ok=True)
  # Grids
  # > Time
  t = system.get_tgrid(**inputs["grids"]["t"])
  # > Density space (lambda)
  rho = np.geomspace(**inputs["grids"]["rho"])
  rho, w_rho = utils.get_gl_quad_1d(rho, deg=2, adim=True)
  sqrt_w_rho = np.sqrt(w_rho)
  # > Initial conditions space (mu)
  mu = tuple([
    np.geomspace(**inputs["grids"]["mu"]["Tint"]),
    np.linspace(**inputs["grids"]["mu"]["X_a"])
  ])

  # Model reduction
  # ---------------
  X, Y = [], []
  for (i, rhoi) in tqdm(enumerate(rho), ncols=80, desc="Densities"):
    # > Linear operators
    lin_ops = system.compute_lin_fom_ops(
      mu=mu,
      rho=rhoi,
      max_mom=int(inputs["max_mom"])
    )
    btrunc = BalancedTruncation(
      operators=lin_ops,
      path_to_saving=path_to_saving,
      saving=False,
      verbose=False
    )
    # > Gramians
    Xi, Yi = btrunc(
      t=t,
      xnot=[0],
      compute_modes=False
    )
    wi = sqrt_w_rho[i]
    X.append(wi*Xi), Y.append(wi*Yi)
  # > Compute balancing modes
  btrunc(
    X=np.hstack(X),
    Y=np.hstack(Y),
    compute_modes=True
  )

  # Copy input file
  # ---------------
  filename = path_to_saving + "/inputs.json"
  with open(filename, "w") as file:
    json.dump(inputs, file, indent=2)
