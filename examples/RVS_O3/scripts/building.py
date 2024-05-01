#!/home/zanardi/.conda/envs/sciml/bin/python
# -*- coding: utf-8 -*-

# Environment
# =====================================
import sys
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
  nb_Tint = 10
  # > Moments of the distribution (molecule)
  max_mom = 10
  # Paths
  paths = {
    "dtb": "./../database/",
    "data": "./data/"
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
  # Linear operators
  Tint = np.geomspace(*Tint_lim, num=nb_Tint)
  lin_ops = model.compute_lin_fom_ops(T=T, Tint=Tint, max_mom=max_mom)
  # Model reduction
  btrunc = BalancedTruncation(
    operators=lin_ops,
    lg_deg=3,
    path_to_saving=paths["data"]
  )
  btrunc(t=t, real_only=True)
