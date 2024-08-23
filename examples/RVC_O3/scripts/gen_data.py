"""
Generate FOM data.
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
import dill as pickle

from ronek import utils
from ronek import systems as sys_mod

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

  # Data generation
  # -----------------------------------
  # Path to saving
  path_to_save = inputs["paths"]["saving"] + "/data/"
  os.makedirs(path_to_save, exist_ok=True)
  # Time grid
  t = system.get_tgrid(**inputs["grids"]["t"])

  # Sampled cases
  # ---------------
  # Construct design matrix
  mu = system.construct_design_mat(**inputs["param_space"]["sampled"])
  # Generate data
  utils.generate_case_parallel(
    sol_fun=system.compute_sol,
    sol_kwargs=dict(
      t=t,
      mu=mu,
      path=path_to_save,
      filename=None
    ),
    nb_samples=inputs["param_space"]["sampled"]["nb_samples"],
    nb_workers=inputs["param_space"]["nb_workers"]
  )
  # Save parameters
  filename = path_to_save + "/mu.p"
  pickle.dump(mu, open(filename, "wb"))

  # Defined cases
  # ---------------
  for (k, mui) in inputs["param_space"]["defined"].items():
    system.compute_sol(
      t=t,
      mu=mui,
      path=None,
      index=None,
      filename=path_to_save + f"/case_{k}.p"
    )

  # Copy input file
  filename = path_to_save + "/inputs.json"
  with open(filename, "w") as file:
    json.dump(inputs, file, indent=2)
