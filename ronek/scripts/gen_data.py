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

from ronek import utils
from ronek import systems as sys_mod

# Main
# =====================================
if (__name__ == '__main__'):

  print("Initialization ...")

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
  path_to_saving = inputs["paths"]["saving"]
  os.makedirs(path_to_saving, exist_ok=True)
  # Time grid
  t = np.geomspace(**inputs["grids"]["t"])

  # Sampled cases
  # ---------------
  # Construct design matrix
  T, mu = system.construct_design_mat(**inputs["param_space"]["sampled"])
  nb_samples_mu = len(mu)
  nb_samples_temp = len(T)
  # Generate data
  si = 0
  ei = nb_samples_mu
  runtime = 0.0
  print("Looping over sampled temperatures:")
  for Ti in T.values.reshape(-1):
    print("> T = %.4e K" % Ti)
    system.update_fom_ops(Ti)
    runtime += utils.generate_case_parallel(
      sol_fun=system.compute_fom_sol,
      range=[si,ei],
      sol_kwargs=dict(
        T=Ti,
        t=t,
        mu=mu.values,
        update=False,
        path=path_to_saving,
        filename=None
      ),
      nb_workers=inputs["param_space"]["nb_workers"]
    )
    si += nb_samples_mu
    ei += nb_samples_mu
  # Save parameters
  for (name, df) in (
    ("mu", mu),
    ("T", T)
  ):
    df.to_csv(
      path_to_saving + f"/samples_{name}.csv",
      float_format="%.8e",
      index=True
    )
  # Save runtime
  runtime /= nb_samples_temp
  with open(path_to_saving + "/runtime.txt", "w") as file:
    file.write("Mean running time: %.8e s" % runtime)

  # Defined cases
  # ---------------
  for (k, param) in inputs["param_space"]["defined"]["cases"].items():
    runtime = system.compute_fom_sol(
      T=float(param["T"]),
      t=t,
      mu=param["mu"],
      mu_type=inputs["param_space"]["defined"].get("mu_type", "mass"),
      update=True,
      filename=path_to_saving + f"/case_{k}.p"
    )
    if (runtime is None):
      print(f"Case '{k}' not converged!")

  # Copy input file
  # ---------------
  filename = path_to_saving + "/inputs.json"
  with open(filename, "w") as file:
    json.dump(inputs, file, indent=2)

  print("Done!")
