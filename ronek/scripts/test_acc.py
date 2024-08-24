"""
Test accuracy of ROM model.
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
from silx.io.dictdump import dicttoh5, h5todict

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
  # Initialization
  # ---------------
  # Path to saving
  path_to_save = inputs["paths"]["saving"] + "/error/" + inputs["eval_err_on"]
  os.makedirs(path_to_save, exist_ok=True)
  # Time grid
  t = utils.load_case(path=inputs["data"]["path"], index=0, key="t")

  # ROM models
  bt_bases = h5todict(inputs["paths"]["bases"])
  bt_bases = [bt_bases[k] for k in ("phi", "psi")]
  cg_model = CoarseGraining(
    T=system.T,
    molecule=path_to_dtb+"/species/molecule.json"
  )

  # Util functions
  # ---------------
  def compute_err_parallel(model="bt"):
    err = jl.Parallel(inputs["data"]["nb_workers"])(
      jl.delayed(system.compute_rom_sol)(
        model=model,
        path=inputs["data"]["path"],
        index=i,
        filename=None,
        eval_err_on=inputs["eval_err_on"]
      ) for i in tqdm(
        iterable=range(inputs["data"]["nb_samples"]),
        ncols=80,
        desc="  Cases",
        file=sys.stdout
      )
    )
    return np.vstack(err)

  def compute_err_stats(err):
    return {
      "t": t,
      "over_t": {"mean": np.mean(err, axis=0), "std": np.std(err, axis=0)},
      "global": {"mean": np.mean(err), "std": np.std(err)}
    }

  def save_err_stats(model, stats):
    dicttoh5(
      treedict=stats,
      h5file=path_to_save + f"/{model}_rom.hdf5",
      overwrite_data=True
    )

  # Loop over ROM dimensions
  # ---------------
  bt_err, cg_err = {}, {}
  for r in range(*inputs["rom_range"]):
    # Solve BT ROM
    print(f"\n> Solving BT ROM with {r} dimensions ...")
    system.update_rom_ops(phi=bt_bases[0][:,:r], psi=bt_bases[1][:,:r])
    errors = compute_err_parallel("bt")
    bt_err[str(r)] = compute_err_stats(errors)
    # Solve CG ROM
    print(f"> Solving CG ROM with {r} dimensions ...")
    system.update_rom_ops(*cg_model(nb_bins=r))
    errors = compute_err_parallel("cg")
    cg_err[str(r)] = compute_err_stats(errors)
  # Save error statistics
  save_err_stats("bt", bt_err)
  save_err_stats("cg", cg_err)

  # Copy input file
  # ---------------
  filename = path_to_save + "/inputs.json"
  with open(filename, "w") as file:
    json.dump(inputs, file, indent=2)
