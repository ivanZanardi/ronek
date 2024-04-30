import sys
sys.path.append("./../")

# Environment
# =====================================
from src import env
env.set(
  backend="torch",
  device="cuda",
  device_idx=1,
  nb_threads=16,
  epsilon=1e-10,
  floatx="float64"
)

# Libraries
# =====================================
import torch
import numpy as np

from src import bpod
from src import backend as bkd
from src.systems import TAFASystem

# Inputs
# =====================================
# System
T = 1e4
# > Initial internal temperature (molecule)
Tint_lim = [3e2, 1e4]
nb_Tint = 10
# > Equilibrium pressure (atom)
p_a_lim = [1e2, 1e5]
nb_p_a = 10
# > Moments of the distribution (molecule)
max_mom = 10
# Paths
paths = {
  "dtb": "./../database/VS_O3_O4/",
  "data": "./../data/"
}

# Isothermal master equation model
# =====================================
model = TAFASystem(
  rates=paths["dtb"]+"kinetics.hdf5",
  species={k: paths["dtb"]+f"species/{k}.json" for k in ("atom", "molecule")},
  use_einsum=False
)
model.update_fom_ops(T)

# Balanced POD
# =====================================
# Time grid
t = np.geomspace(1e-12, 1e-2, num=50)
t = np.insert(t, 0, 0.0)
# Internal temperature and pressure grids
Tint = np.linspace(*Tint_lim, num=nb_Tint)
p_a = np.linspace(*p_a_lim, num=nb_p_a)
# Loop over pressures
X, Y = [], []
for p in p_a:
  lin_ops = model.compose_lin_fom_ops(p_a=p, T=T, Tint=Tint, max_mom=max_mom)
  # Model reduction
  bpod_lin = bpod.Linear(
    operators=lin_ops,
    lg_deg=3,
    path_to_save=paths["data"]+f"/bpod_lin_tafa.T{int(T)}K.m{max_mom}",
    saving=False
  )
  bpod_lin(
    t=t,
    real_only=True,
    maxrank=0,
    compute_modes=False
  )
  X.append(bpod_lin.X)
  Y.append(bpod_lin.Y)

print("Computing the balancing modes ...")
bpod_lin.saving = True
X, Y = torch.hstack(X), torch.hstack(Y)
bpod_lin.compute_balancing_modes(X, Y)
