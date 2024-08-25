"""
Visualize ROM vs FOM trajectories.
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
import matplotlib.pyplot as plt
plt.style.use(inputs["plot"].get("style", None))

from tqdm import tqdm
from ronek import utils
from ronek import postproc as pp
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
  # Initialization
  # ---------------
  # Path to saving
  path_to_saving = inputs["paths"]["saving"] + "/visual/"
  os.makedirs(path_to_saving, exist_ok=True)

  # ROM models
  bt_bases = h5todict(inputs["paths"]["bases"])
  bt_bases = [bt_bases[k] for k in ("phi", "psi")]
  cg_model = CoarseGraining(
    T=system.T,
    molecule=path_to_dtb+"/species/molecule.json"
  )

  # Loop over test cases
  # ---------------
  for case_id in inputs["data"]["case_ids"]:
    print(f"\nSolving case '{case_id}' ...")
    # > Load test case
    filename = inputs["data"]["path"]+f"/case_{case_id}.p"
    icase = utils.load_case(filename=filename)
    n_fom, t, n0 = [icase[k] for k in ("n", "t", "n0")]
    # > Loop over ROM dimensions
    for r in range(*inputs["rom_range"]):
      # > Saving folder
      path_to_saving_i = path_to_saving + f"/case_{case_id}/r{r}/"
      os.makedirs(path_to_saving_i, exist_ok=True)
      # > Solve BT ROM
      print(f"> Solving BT ROM with {r} dimensions ...")
      system.update_rom_ops(phi=bt_bases[0][:,:r], psi=bt_bases[1][:,:r])
      n_rom_bt = system.solve_rom_bt(t, n0)
      # > Solve CG ROM
      print(f"> Solving CG ROM with {r} dimensions ...")
      system.update_rom_ops(*cg_model(nb_bins=r))
      n_rom_cg = system.solve_rom_cg(t, n0)
      # > Collect solutions
      sols = {
        "FOM-StS": n_fom[1],
        "ROM-BT": n_rom_bt[1],
        "ROM-CG": n_rom_cg[1]
      }
      # > Postprocessing
      print(f"> Postprocessing with {r} dimensions ...")
      common_kwargs = dict(
        path=path_to_saving_i,
        t=t,
        n_m=sols,
        molecule=system.species["molecule"]
      )
      pp.plot_mom_evolution(
        max_mom=2,
        **common_kwargs
      )
      pp.plot_multi_dist_2d(
        teval=inputs["data"]["teval"][case_id],
        markersize=inputs["plot"]["markersize"],
        **common_kwargs
      )
      if inputs["plot"]["animate"]:
        pp.animate_dist(
          markersize=inputs["plot"]["markersize"],
          **common_kwargs
        )

  # Copy input file
  # ---------------
  filename = path_to_saving + "/inputs.json"
  with open(filename, "w") as file:
    json.dump(inputs, file, indent=2)
