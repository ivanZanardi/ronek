"""
Evaluate accuracy of ROM model.
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
import matplotlib.pyplot as plt
plt.style.use(inputs["plot"].get("style", None))

from tqdm import tqdm
from ronek import utils
from ronek import postproc as pp
from ronek import systems as sys_mod
from ronek.roms import CoarseGrainingM0
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
  path_to_saving = inputs["paths"]["saving"]+"/error/"+inputs["eval_err_on"]
  os.makedirs(path_to_saving, exist_ok=True)
  # Time grid
  t = utils.load_case(path=inputs["data"]["path"], index=0, key="t")

  # ROM models
  # > Balanced truncation (BT) / Petrov-Galerkin (PG)
  bt_bases = h5todict(inputs["paths"]["bases"])
  bt_bases = [bt_bases[k] for k in ("phi", "psi")]
  # > Coarse graining (CG)
  cg_model = inputs.get("cg_model", {"active": False})
  if cg_model["active"]:
    cg_m0 = CoarseGrainingM0(
      molecule=path_to_dtb+"/species/molecule.json", T=system.T
    )

  # Util functions
  # ---------------
  def compute_err_parallel():
    err = jl.Parallel(inputs["data"]["nb_workers"])(
      jl.delayed(system.compute_rom_sol)(
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
    if (inputs["eval_err_on"] == "mom"):
      return np.stack(err, axis=0)
    else:
      return np.vstack(err)

  def compute_err_stats(err):
    return {
      "t": t,
      "mean": np.mean(err, axis=0),
      "std": np.std(err, axis=0)
    }

  def save_err_stats(model, stats):
    dicttoh5(
      treedict=stats,
      h5file=path_to_saving + f"/{model}_rom.hdf5",
      overwrite_data=True
    )

  # Loop over ROM dimensions
  # ---------------
  bt_err, cg_err = {}, {}
  for r in range(*inputs["rom_range"]):
    # Solve PG ROM
    print(f"\n> Solving PG ROM with {r} dimensions ...")
    system.update_rom_ops(phi=bt_bases[0][:,:r], psi=bt_bases[1][:,:r])
    if (not system.rom_valid):
      print(f"  PG ROM with {r} dimensions not valid!")
      continue
    errors = compute_err_parallel()
    bt_err[str(r)] = compute_err_stats(errors)
    # Solve CG ROM
    if cg_model["active"]:
      print(f"> Solving CG ROM with {r} dimensions ...")
      cg_m0.build(nb_bins=r)
      system.update_rom_ops(cg_m0.phi, cg_m0.psi)
      errors = compute_err_parallel()
      cg_err[str(r)] = compute_err_stats(errors)
  # Save/plot error statistics
  common_kwargs = dict(
    eval_err_on=inputs["eval_err_on"],
    err_scale=inputs["plot"].get("err_scale", "linear"),
    molecule_label=inputs["plot"]["molecule_label"],
    subscript=inputs["plot"].get("subscript", "i"),
    max_mom=inputs["plot"].get("max_mom", 2)
  )
  save_err_stats("bt", bt_err)
  pp.plot_err_evolution(
    path=path_to_saving+"/bt/",
    err=bt_err,
    **common_kwargs
  )
  if cg_model["active"]:
    save_err_stats("cg", cg_err)
    pp.plot_err_evolution(
      path=path_to_saving+"/cg/",
      err=cg_err,
      **common_kwargs
    )

  # Copy input file
  # ---------------
  filename = path_to_saving + "/inputs.json"
  with open(filename, "w") as file:
    json.dump(inputs, file, indent=2)
