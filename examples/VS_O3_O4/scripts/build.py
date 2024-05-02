#!/home/zanardi/.conda/envs/sciml/bin/python
# -*- coding: utf-8 -*-

# Environment
# =====================================
import sys
import importlib
if (importlib.util.find_spec("ronek") is None):
  sys.path.append("./../../../ronek/")

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
from ronek.bpod import BPOD


if (__name__ == '__main__'):

  # Inputs
  # ===================================
  # System
  T = 1e4
  # > Initial internal temperature (molecule)
  Tint_grid = {
    "lim": [3e2, 1e4],
    "num": 20
  }
  # > Equilibrium pressure (atom)
  p_grid = {
    "lim": [1e3, 1e5],
    "num": 20
  }
  # > Moments of the distribution (molecule)
  max_mom = 10 # 2 10
  # Paths
  paths = {
    "dtb": "./../database/",
    "data": f"./data/max_mom_{max_mom}/"
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
  t = np.geomspace(1e-12, 1e-2, num=50)
  t = np.insert(t, 0, 0.0)
  # Pressure and internal temperature grids
  p = np.geomspace(*p_grid["lim"], num=p_grid["num"])
  Tint = np.geomspace(*Tint_grid["lim"], num=Tint_grid["num"])
  # Loop over pressures
  X, Y = [], []
  for pi in tqdm(p, ncols=80, desc="Pressures"):
    # Model reduction
    btrunc = BPOD(
      operators=model.compute_lin_fom_ops(
        p=pi, T=T, Tint=Tint, max_mom=max_mom
      ),
      lg_deg=3,
      path_to_saving=paths["data"],
      saving=False,
      verbose=False
    )
    Xi, Yi = btrunc(t=t, compute_modes=False)
    X.append(Xi), Y.append(Yi)
  # Compute balancing modes
  X, Y = np.hstack(X), np.hstack(Y)
  btrunc.compute_balancing_modes(X, Y)
