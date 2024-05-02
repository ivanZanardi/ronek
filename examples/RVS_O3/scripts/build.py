#!/home/zanardi/.conda/envs/sciml/bin/python
# -*- coding: utf-8 -*-

# Environment
# =====================================
import sys
import importlib
if (importlib.util.find_spec("ronek") is None):
  sys.path.append("./../../../")

from ronek import env
env.set(
  device="cuda",
  device_idx=1,
  nb_threads=16,
  floatx="float64"
)

# Libraries
# =====================================
import numpy as np

from ronek.systems import TASystem
from ronek.bal_trunc import BalancedTruncation


if (__name__ == '__main__'):

  # Inputs
  # ===================================
  # System
  T = 1e4
  # > Initial internal temperature (molecule)
  Tint_lim = [2e2, 1e4]
  nb_Tint = 50
  # > Moments of the distribution (molecule)
  max_mom = 0
  # Paths
  paths = {
    "dtb": "./../database/",
    "data": f"./data/max_mom_{max_mom}/"
  }

  # Isothermal master equation model
  # ===================================
  model = TASystem(
    rates=paths["dtb"] + "/kinetics.hdf5",
    species={
      k: paths["dtb"] + f"/species/{k}.json" for k in ("atom", "molecule")
    },
    use_einsum=False
  )
  model.update_fom_ops(T)

  # Balanced truncation
  # ===================================
  # Time grid
  t = np.geomspace(1e-13, 1e-1, num=100)
  t = np.insert(t, 0, 0.0)
  # Internal temperature grid
  Tint = np.geomspace(*Tint_lim, num=nb_Tint)
  # Model reduction
  btrunc = BalancedTruncation(
    operators=model.compute_lin_fom_ops(T=T, Tint=Tint, max_mom=max_mom),
    lg_deg=3,
    path_to_saving=paths["data"]
  )
  btrunc(t)
