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
  # Path to saving
  path_to_saving = inputs["paths"]["saving"] + "/visual/"
  os.makedirs(path_to_saving, exist_ok=True)

  # ROM models
  bt_bases = h5todict(inputs["paths"]["bases"])
  bt_bases = [bt_bases[k] for k in ("phi", "psi")]
  cg_model = CoarseGraining(molecule=path_to_dtb+"/species/molecule.json")

  # Loop over test cases
  for case_id in tqdm(
    iterable=inputs["data"]["case_ids"],
    ncols=80,
    desc="  Cases",
    file=sys.stdout
  ):
    # > Load test case
    icase = utils.load_case(inputs["data"]["path"]+f"/case_{case_id}.p")
    n_fom, t, n0 = [icase[k] for k in ("n", "t", "n0")]
    # > Loop over ROM dimensions
    for r in range(*inputs["rom_range"]):
      # > Saving folder
      path_to_saving_i = path_to_saving + f"/case_{case_id}/r{r}/"
      os.makedirs(path_to_saving_i, exist_ok=True)
      # > Solve BT ROM
      print(f"\n> Solving BT ROM with {r} dimensions ...")
      system.update_rom_ops(phi=bt_bases[0][:,:r], psi=bt_bases[1][:,:r])
      n_rom_bt = system.solve_rom_bt(t, n0)
      # > Solve CG ROM
      system.update_rom_ops(*cg_model(nb_bins=r))
      n_rom_cg = system.solve_rom_cg(t, n0)
      # > Collect solutions
      sols = {
        "FOM-StS": n_fom[1],
        "ROM-BT": n_rom_bt[1],
        "ROM-CG": n_rom_cg[1]
      }
      # > Postprocessing
      pp.plot_mom_evolution(
        path=path_to_saving_i,
        t=t,
        n_m=sols,
        molecule=system.molecule,
        max_mom=2
      )
      pp.plot_multi_dist_2d(
        path=path_to_saving_i,
        teval=inputs["data"]["teval"][case_id],
        t=t,
        n_m=sols,
        molecule=system.molecule,
        markersize=1
      )
      if inputs["animate"]:
        pp.animate_dist(
          path=path_to_saving_i,
          t=t,
          n_m=sols,
          molecule=system.molecule,
          markersize=1
        )

  # Copy input file
  # ---------------
  filename = path_to_saving + "/inputs.json"
  with open(filename, "w") as file:
    json.dump(inputs, file, indent=2)
