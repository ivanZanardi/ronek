"""
Build balanced truncation-based ROM.
"""

import os
import sys
import json
import time
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

  runtime = time.time()

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
  # Path to saving
  path_to_saving = inputs["paths"]["saving"]
  os.makedirs(path_to_saving, exist_ok=True)

  # Quadrature points
  quad = {}
  # > Time
  t, w_t = utils.get_gl_quad_1d(
    x=system.get_tgrid(**inputs["grids"]["t"]),
    deg=2,
    dist="uniform"
  )
  quad["t"] = {"x": t, "w": np.sqrt(w_t)}
  # > Initial conditions space (mu)
  mu, w_mu = utils.get_gl_quad_2d(
    x=np.geomspace(**inputs["grids"]["mu"]["T0"]),
    y=np.linspace(**inputs["grids"]["mu"]["w0_a"]),
    deg=2,
    dist_x="loguniform",
    dist_y="uniform"
  )
  quad["mu"] = {"x": mu, "w": np.sqrt(w_mu)}
  # > Density space (lambda)
  rho, w_rho = utils.get_gl_quad_1d(
    x=np.geomspace(**inputs["grids"]["rho"]),
    deg=2,
    dist="uniform"
  )
  quad["rho"] = {"x": rho, "w": np.sqrt(w_rho)}

  # Model reduction
  # ---------------
  X, Y = [], []
  for (i, rhoi) in enumerate(
    tqdm(quad["rho"]["x"], ncols=80, desc="Densities")
  ):
    # > Linear operators
    lin_ops = system.compute_lin_fom_ops(
      mu=quad["mu"]["x"],
      rho=rhoi,
      max_mom=int(inputs["max_mom"])
    )
    btrunc = BalancedTruncation(
      operators=lin_ops,
      quadrature=quad,
      path_to_saving=path_to_saving,
      saving=False,
      verbose=False
    )
    # > Gramians
    Xi, Yi = btrunc(
      xnot=[0],
      compute_modes=False
    )
    wi = quad["rho"]["w"][i]
    X.append(wi*Xi), Y.append(wi*Yi)

  # > Compute balancing modes
  btrunc.verbose = True
  btrunc(
    X=np.hstack(X),
    Y=np.hstack(Y),
    compute_modes=True,
    runtime=btrunc.runtime
  )

  # Copy input file
  # ---------------
  filename = path_to_saving + "/inputs.json"
  with open(filename, "w") as file:
    json.dump(inputs, file, indent=2)

  btrunc.runtime["tot"] = time.time()-runtime
  filename = path_to_saving + "/runtime.json"
  with open(filename, "w") as file:
    json.dump(btrunc.runtime, file, indent=2)
