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
  device_idx=0,
  nb_threads=16,
  floatx="float64"
)

# Libraries
# =====================================
import numpy as np

from ronek.systems import TASystem
from ronek.roms import BalancedTruncation


if (__name__ == '__main__'):

  # Inputs
  # ===================================
  # Time
  t_grid = {
    "lim": [1e-12, 1e-2],
    "pts": 50
  }
  # System
  T = 1e4
  # > Initial internal temperature (molecule)
  Tint_grid = {
    "lim": [3e2, 1e4],
    "pts": 20
  }
  # > Moments of the distribution (molecule)
  max_mom = 10
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
  model.set_eq_ratio(T)

  # Balanced truncation
  # ===================================
  # Time grid
  t = model.get_tgrid(t_grid["lim"], t_grid["pts"])
  # Internal temperature grid
  Tint = np.geomspace(*Tint_grid["lim"], num=Tint_grid["pts"])
  # Model reduction
  btrunc = BalancedTruncation(
    operators=model.compute_lin_fom_ops(Tint=Tint, max_mom=max_mom),
    lg_deg=3,
    path_to_saving=paths["data"]
  )
  btrunc(t)
