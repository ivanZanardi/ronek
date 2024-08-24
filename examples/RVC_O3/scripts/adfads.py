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
import numpy as np
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
  # Initialization
  # ---------------
  # Path to saving
  path_to_saving = inputs["paths"]["saving"] + "/data/"
  os.makedirs(path_to_saving, exist_ok=True)
  # Time grid
  t = np.geomspace(**inputs["grids"]["t"])

  # Sampled cases
  # ---------------
  # Construct design matrix
  mu = system.construct_design_mat(**inputs["param_space"]["sampled"])
  # Generate data
  utils.generate_case_parallel(
    sol_fun=system.compute_fom_sol,
    sol_kwargs=dict(
      t=t,
      mu=mu,
      path=path_to_saving,
      filename=None
    ),
    nb_samples=inputs["param_space"]["sampled"]["nb_samples"],
    nb_workers=inputs["param_space"]["nb_workers"]
  )
  # Save parameters
  filename = path_to_saving + "/mu.p"
  pickle.dump(mu, open(filename, "wb"))

  # Defined cases
  # ---------------
  for (k, mui) in inputs["param_space"]["defined"].items():
    system.compute_fom_sol(
      t=t,
      mu=mui,
      path=None,
      index=None,
      filename=path_to_saving + f"/case_{k}.p"
    )


t = model.get_tgrid(tgrid["lim"], tgrid["num"])
for (name, (T, p, Xa)) in ic.items():
  y0 = model.get_init_sol(T, p, Xa)
  # Solving
  print(f"> Solving FOM for '{name}' test case ...")
  yfom = model.solve_fom(t, y0, rtol=1e-6)
  for r in rom_dims:
    model.update_rom_ops(phi[:,:r], psi[:,:r])
    print(f"  > Solving ROM for '{name}' test case with {r} dimensions ...")
    yrom = model.solve_rom_bt(t, y0, rtol=1e-6, use_abs=False)
    # Postprocessing
    print(f"  > Postprocessing FOM and ROM solutions ...")
    path = paths["data"] + f"/figs/sol/{name}_R{r}/"
    os.makedirs(path, exist_ok=True)
    # > Moments
    plot_moments(path, t, yfom[1], yrom[1], ei.reshape(-1,1), max_mom=2)
    # > Distribution static
    plot_dist(path, teval[name], t, yfom[1], yrom[1], ei, gi)
    if (max_mom > 2):
      # > Distribution dynamic
      animate_dist(path, t, yfom[1], yrom[1], ei, gi)




  # Copy input file
  # ---------------
  filename = path_to_saving + "/inputs.json"
  with open(filename, "w") as file:
    json.dump(inputs, file, indent=2)
