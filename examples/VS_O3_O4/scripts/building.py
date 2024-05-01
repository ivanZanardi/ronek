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

from tqdm import tqdm
from ronek.systems import TAFASystem
from ronek.bal_trunc import BalancedTruncation


if (__name__ == '__main__'):

  # Inputs
  # ===================================
  # System
  T = 1e4
  # > Initial internal temperature (molecule)
  Tint_lim = [2e2, 1e4]
  nb_Tint = 10
  # > Equilibrium pressure (atom)
  p_lim = [1e3, 1e5]
  nb_p = 10
  # Paths
  paths = {
    "dtb": "./../database/",
    "data": "./data/"
  }

  # Isothermal master equation model
  # ===================================
  model = TAFASystem(
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
  # Pressure and internal temperature grids
  p = np.geomspace(*p_lim, num=nb_p)
  Tint = np.geomspace(*Tint_lim, num=nb_Tint)
  # Loop over pressures
  X, Y = [], []
  for pi in tqdm(p, ncols=80, desc="Pressures"):
    # Model reduction
    btrunc = BalancedTruncation(
      operators=model.compute_lin_fom_ops(p=pi, T=T, Tint=Tint),
      lg_deg=3,
      path_to_saving=paths["data"],
      saving=False,
      verbose=False
    )
    Xi, Yi = btrunc(t=t, real_only=True, compute_modes=False)
    X.append(Xi), Y.append(Yi)
  # Compute balancing modes
  X, Y = np.hstack(X), np.hstack(Y)
  btrunc.compute_balancing_modes(X, Y)
