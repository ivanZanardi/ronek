import os
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

# Use custom style
plt.style.use(
  "/home/zanardi/Workspace/Research/styles/matplotlib/paper_1column.mplstyle"
)

from ronek.postproc import plotting as pltt
from ronek.postproc import animation as anim


# Plotting
# =====================================
def plot_moments(path, t, yfom, yrom, e, max_mom=2):
  path = path + "/moments/"
  os.makedirs(path, exist_ok=True)
  for m in range(max_mom):
    mom_fom = np.sum(yfom * e**m, axis=0)
    mom_rom = np.sum(yrom * e**m, axis=0)
    if (m == 0):
      yscale = "log"
      label_sol = r"$n$ [m$^{-3}$]"
      label_err = r"$\Delta n$ [\%]"
      mom0_fom, mom0_rom = mom_fom, mom_rom
    else:
      yscale = "linear"
      if (m == 1):
        label_sol = r"$e_{\text{int}}$ [eV]"
        label_err = r"$\Delta e_{\text{int}}$ [\%]"
      else:
        label_sol = fr"$\gamma_{m}$ [eV$^{m}$]"
        label_err = fr"$\Delta\gamma_{m}$ [\%]"
      mom_fom /= mom0_fom
      mom_rom /= mom0_rom
    # Plot moment
    pltt.evolution(
      x=t[1:],
      y=mom_fom[1:],
      y_pred=mom_rom[1:],
      labels=[r"$t$ [s]", label_sol],
      scales=["log", yscale],
      figname=path + f"/m{m}",
      save=True,
      show=False
    )
    # Plot moment error
    mom_err = 100 * np.abs(mom_rom - mom_fom) / np.abs(mom_fom)
    pltt.evolution(
      x=t[1:],
      y=mom_err[1:],
      labels=[r"$t$ [s]", label_err],
      scales=['log', 'linear'],
      figname=path + f"/m{m}_err",
      save=True,
      show=False
    )

def plot_dist(path, teval, t, yfom, yrom, e, g, markersize=6):
  path = path + "/dist/"
  os.makedirs(path, exist_ok=True)
  yfom = sp.interpolate.interp1d(t, yfom, kind='cubic')(teval)
  yrom = sp.interpolate.interp1d(t, yrom, kind='cubic')(teval)
  for i in range(len(teval)):
    pltt.dist_2d(
      x=e,
      y=yfom[:,i]/g,
      y_pred=yrom[:,i]/g,
      labels=[r"$\epsilon_i$ [eV]", r"$n_i/g_i$ [m$^{-3}$]"],
      scales=["linear", "log"],
      markersize=markersize,
      figname=path + f"/t{i+1}",
      save=True,
      show=False
    )

def animate_dist(path, t, yfom, yrom, e, g, markersize=6):
  y = {
    "FOM": yfom.T[1:] / g.reshape(1,-1),
    "ROM": yrom.T[1:] / g.reshape(1,-1)
  }
  anim.animate(
    t=t[1:],
    x=e,
    y=y,
    markersize=markersize,
    frames=100,
    fps=10,
    filename=path + "/dist.mp4",
    dpi=600,
    save=True,
    show=False
  )
