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
  path_to_saving = inputs["paths"]["saving"]+"/error/"+inputs["eval_err"]
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
    iterable = tqdm(
      iterable=range(inputs["data"]["nb_samples"]),
      ncols=80,
      desc="  Cases",
      file=sys.stdout
    )
    nb_workers = inputs["data"]["nb_workers"]
    if (nb_workers > 1):
      sol = jl.Parallel(inputs["data"]["nb_workers"])(
        jl.delayed(system.compute_rom_sol)(
          path=inputs["data"]["path"],
          index=i,
          filename=None,
          eval_err=inputs["eval_err"]
        ) for i in iterable
      )
    else:
      sol = [
        system.compute_rom_sol(
          path=inputs["data"]["path"],
          index=i,
          filename=None,
          eval_err=inputs["eval_err"]
        ) for i in iterable
      ]
    err, runtime = list(zip(*sol))
    if (inputs["eval_err"] == "mom"):
      return np.stack(err, axis=0), runtime
    else:
      return np.vstack(err), runtime

  def compute_err_stats(err):
    return {
      "t": t,
      "mean": np.mean(err, axis=0),
      "std": np.std(err, axis=0)
    }

  def compute_runtime_stats(runtime):
    return {
      "mean": np.mean(runtime),
      "std": np.std(runtime)
    }

  def save_err_stats(model, stats):
    dicttoh5(
      treedict=stats,
      h5file=path_to_saving + f"/{model}_err.hdf5",
      overwrite_data=True
    )

  def save_runtime_stats(model, stats):
    filename = path_to_saving + f"/{model}_runtime.json"
    with open(filename, "w") as file:
      json.dump(stats, file, indent=2)

  # Loop over ROM dimensions
  # ---------------
  bt_err, bt_runtime = {}, {}
  cg_err, cg_runtime = {}, {}
  for r in range(*inputs["rom_range"]):
    r_str = str(r)
    # Solve PG ROM
    print(f"\n> Solving PG ROM with {r} dimensions ...")
    system.update_rom_ops(phi=bt_bases[0][:,:r], psi=bt_bases[1][:,:r])
    errors, runtime = compute_err_parallel()
    bt_err[r_str] = compute_err_stats(errors)
    bt_runtime[r_str] = compute_runtime_stats(runtime)
    # Solve CG ROM
    if cg_model["active"]:
      print(f"> Solving CG ROM with {r} dimensions ...")
      cg_m0.build(nb_bins=r)
      system.update_rom_ops(cg_m0.phi, cg_m0.psi)
      errors, runtime = compute_err_parallel()
      cg_err[r_str] = compute_err_stats(errors)
      cg_runtime[r_str] = compute_runtime_stats(runtime)
  # Save/plot error statistics
  common_kwargs = dict(
    eval_err=inputs["eval_err"],
    hline=inputs["plot"].get("hline", None),
    err_scale=inputs["plot"].get("err_scale", "linear"),
    molecule_label=inputs["plot"]["molecule_label"],
    subscript=inputs["plot"].get("subscript", "i"),
    max_mom=inputs["plot"].get("max_mom", 2)
  )
  print("\n> Saving PG ROM error evolution ...")
  save_err_stats("bt", bt_err)
  print("\n> Plotting PG ROM error evolution ...")
  pp.plot_err_evolution(
    path=path_to_saving+"/bt/",
    err=bt_err,
    **common_kwargs
  )
  if cg_model["active"]:
    print("\n> Saving CG ROM error evolution ...")
    save_err_stats("cg", bt_err)
    print("\n> Plotting CG ROM error evolution ...")
    pp.plot_err_evolution(
      path=path_to_saving+"/cg/",
      err=cg_err,
      **common_kwargs
    )

  # Save running times
  save_runtime_stats("bt", bt_runtime)
  if cg_model["active"]:
    save_runtime_stats("cg", cg_runtime)

  # Copy input file
  # ---------------
  filename = path_to_saving + "/inputs.json"
  with open(filename, "w") as file:
    json.dump(inputs, file, indent=2)
