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

from ronek import roms
from ronek import utils
from ronek import postproc as pp
from ronek import systems as sys_mod
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
  # > Balanced truncation (BT)
  bt_bases = h5todict(inputs["paths"]["bases"])
  bt_bases = [bt_bases[k] for k in ("phi", "psi")]
  # > Coarse graining (CG)
  cg_model = inputs.get(
    "cg_model", {
      "0": {"active": False},
      "1": {"active": False}
    }
  )
  if cg_model["0"]["active"]:
    cg_m0 = roms.CoarseGrainingM0(
      molecule=path_to_dtb+"/species/molecule.json", T=system.T
    )
  if cg_model["1"]["active"]:
    cg_m1 = roms.CoarseGrainingM1(
      molecule=path_to_dtb+"/species/molecule.json"
    )
  cg_active_both = (cg_model["1"]["active"] + cg_model["0"]["active"] == 2)

  # Loop over test cases
  # ---------------
  for icase in inputs["data"]["cases"]:
    print(f"\nSolving case '{icase}' ...")
    # > Load test case
    filename = inputs["data"]["path"]+f"/case_{icase}.p"
    data = utils.load_case(filename=filename)
    n_fom, t, n0 = [data[k] for k in ("n", "t", "n0")]
    # > Solutions container
    sols = {"FOM-StS": n_fom[1]}
    # > Loop over ROM dimensions
    for r in range(*inputs["rom_range"]):
      # > Saving folder
      path_to_saving_i = path_to_saving + f"/case_{icase}/r{r}/"
      os.makedirs(path_to_saving_i, exist_ok=True)
      # > Solve ROM-PG
      print(f"> Solving ROM-PG with {r} dimensions ...")
      system.update_rom_ops(phi=bt_bases[0][:,:r], psi=bt_bases[1][:,:r])
      if (not system.rom_valid):
        print(f"  PG ROM with {r} dimensions not valid!")
        continue
      n_rom_bt = system.solve_rom(t, n0)
      sols["ROM-PG"] = n_rom_bt[1]
      # > Solve ROM-CG
      if cg_model["0"]["active"]:
        name = "ROM-CG-M0" if cg_active_both else "ROM-CG"
        print(f"> Solving {name} with {r} bins ...")
        cg_m0.build(nb_bins=r)
        system.update_rom_ops(cg_m0.phi, cg_m0.psi)
        n_rom_cg = system.solve_rom(t, n0)
        name = "ROM-CG-M0" if cg_active_both else "ROM-CG"
        sols[name] = n_rom_cg[1]
      if (cg_model["1"]["active"] and (2*cg_model["1"]["nb_bins"] == r)):
        name = "ROM-CG-M1" if cg_active_both else "ROM-CG"
        print(f"> Reading {name} with {int(r/2)} bins ...")
        cg_m1.build(mapping=cg_model["1"]["mapping"])
        sols[name] = cg_m1.decode(
          x=cg_m1.read_sol(
            filename=cg_model["1"]["cases"][icase],
            teval=t
          )
        )
      # > Postprocessing
      print(f"> Postprocessing with {r} dimensions ...")
      common_kwargs = dict(
        path=path_to_saving_i,
        t=t,
        n_m=sols,
        molecule=system.species["molecule"]
      )
      pp.plot_mom_evolution(
        max_mom=inputs["plot"].get("max_mom", 2),
        molecule_label=inputs["plot"]["molecule_label"],
        err_scale=inputs["plot"].get("err_scale", "linear"),
        **common_kwargs
      )
      pp.plot_multi_dist_2d(
        teval=inputs["data"]["teval"][icase],
        markersize=inputs["plot"].get("markersize", 1),
        subscript=inputs["plot"].get("subscript", "i"),
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
