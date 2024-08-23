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
import numpy as np
import joblib as jl

from tqdm import tqdm
from ronek import utils
from ronek import systems as sys_mod
from ronek.roms import CoarseGraining
from silx.io.dictdump import h5todict

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

  # Testing
  # -----------------------------------
  # Path to saving
  path_to_save = inputs["paths"]["save"]
  os.makedirs(path_to_save, exist_ok=True)

  # ROM models
  bt_bases = h5todict(inputs["paths"]["bases"])
  bt_bases = [bt_bases[k] for k in ("phi", "psi")]
  cg_model = CoarseGraining(molecule=path_to_dtb+"/species/molecule.json")

  # Loop over cases
  iterable = tqdm(
    iterable=range(inputs["data"]["nb_samples"]),
    ncols=80,
    desc="Cases",
    file=sys.stdout
  )
  # Arguments for computing ROM solution
  rom_sol_kwargs = dict(
    path=inputs["data"]["path"],
    filename=None,
    eval_err_on=inputs["eval_err_on"]
  )
  # Loop ROM dimensions
  bt_err, cg_err = [], []
  for r in range(*inputs["rom_range"]):
    # Solve BT ROM
    system.update_rom_ops(phi=bt_bases[0][:,:r], psi=bt_bases[1][:,:r])
    errors = jl.Parallel(inputs["data"]["nb_workers"])(
      jl.delayed(system.compute_rom_sol)(
        model="bt", index=i, **rom_sol_kwargs
      ) for i in iterable
    )
    errors = np.vstack(errors)
    bt_err.append((r, np.mean(errors), np.std(errors)))
    # Solve CG ROM
    system.update_rom_ops(*cg_model(nb_bins=r))
    errors = jl.Parallel(inputs["data"]["nb_workers"])(
      jl.delayed(system.compute_rom_sol)(
        model="cg", index=i, **rom_sol_kwargs
      ) for i in iterable
    )
    errors = np.vstack(errors)
    cg_err.append((r, np.mean(errors), np.std(errors)))

  # Copy input file
  filename = path_to_save + "/inputs.json"
  with open(filename, "w") as file:
    json.dump(inputs, file, indent=2)