import os
import numpy as np
import scipy as sp
import matplotlib
import matplotlib.pyplot as plt

from .. import const

COLORS = matplotlib.rcParams["axes.prop_cycle"].by_key()["color"]


# Plotting
# =====================================
# Cumulative energy
def cum_energy(
  y,
  figname=None,
  save=False,
  show=True
):
  # Initialize figure
  fig = plt.figure()
  ax = fig.add_subplot()
  # x axis
  x = np.arange(1,len(y)+1)
  ax.set_xlabel("$i$-th basis")
  # y axis
  ax.set_ylabel("Cumulative energy")
  # Plotting
  ax.plot(x, y, marker="o")
  # Tight layout
  plt.tight_layout()
  if save:
    plt.savefig(figname)
  if show:
    plt.show()
  plt.close()

# Time evolution
def evolution(
  x,
  y,
  labels=[r"$t$ [s]", r"$n$ [m$^{-3}$]"],
  scales=["log", "linear"],
  figname=None,
  save=False,
  show=False
):
  # Initialize figures
  fig = plt.figure()
  ax = fig.add_subplot()
  # x axis
  ax.set_xlabel(labels[0])
  ax.set_xscale(scales[0])
  ax.set_xlim([np.amin(x), np.amax(x)])
  # y axis
  ax.set_ylabel(labels[1])
  ax.set_yscale(scales[1])
  # Plotting
  if isinstance(y, dict):
    i = 0
    for (k, yk) in y.items():
      if ("FOM" in k.upper()):
        c = "k"
        ls = "-"
      else:
        c = COLORS[i]
        ls = "--"
        i += 1
      ax.plot(x, yk, ls=ls, c=c, label=k)
    ax.legend()
  else:
    ax.plot(x, y, "-", c="k")
  # Tight layout
  plt.tight_layout()
  if save:
    plt.savefig(figname)
  if show:
    plt.show()
  plt.close()

def mom_evolution(path, t, yfom, yrom, e, max_mom=2):
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
    evolution(
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
    evolution(
      x=t[1:],
      y=mom_err[1:],
      labels=[r"$t$ [s]", label_err],
      scales=['log', 'linear'],
      figname=path + f"/m{m}_err",
      save=True,
      show=False
    )

# 2D distribution
def dist_2d(
  x,
  y,
  labels=[r"$\epsilon_i$ [eV]", r"$n_i/g_i$ [m$^{-3}$]"],
  scales=["linear", "log"],
  markersize=6,
  figname=None,
  save=False,
  show=False
):
  # Initialize figures
  fig = plt.figure()
  ax = fig.add_subplot()
  # x axis
  ax.set_xlabel(labels[0])
  ax.set_xscale(scales[0])
  # y axis
  ax.set_ylabel(labels[1])
  ax.set_yscale(scales[1])
  # Plotting
  style = dict(
    linestyle="",
    marker="o"
  )
  if isinstance(y, dict):
    i = 0
    lines = []
    for (k, yk) in y.items():
      if ("FOM" in k.upper()):
        c = "k"
      else:
        c = COLORS[i]
        i += 1
      ax.plot(x, yk, c=c, markersize=markersize, **style)
      lines.append(plt.plot([], [], c=c, **style)[0])
    ax.legend(lines, list(y.keys()))
  else:
    ax.plot(x, y, c="k", markersize=markersize, **style)
  # Tight layout
  plt.tight_layout()
  if save:
    plt.savefig(figname)
  if show:
    plt.show()
  plt.close()

def multi_dist_2d(
  path,
  teval,
  t,
  n_m,
  molecule,
  markersize=6
):
  path = path + "/dist/"
  os.makedirs(path, exist_ok=True)
  # Interpolate at "teval" points
  n_m_eval = {}
  for (k, nk) in n_m.items():
    if (nk.shape[-1] != molecule.nb_comp):
      nk = nk.T
    nk = sp.interpolate.interp1d(t, nk, kind="cubic", axis=0)(teval)
    n_m_eval[k] = nk / molecule.lev["g"]
  for i in range(len(teval)):
    dist_2d(
      x=molecule.lev["e"] / const.eV_to_J,
      y={k: n_m_eval[k][i] for k in n_m.keys() if ("FOM" in k)},
      y_pred={k: n_m_eval[k][i] for k in n_m.keys() if ("FOM" not in k)},
      labels=[r"$\epsilon_i$ [eV]", r"$n_i/g_i$ [m$^{-3}$]"],
      scales=["linear", "log"],
      markersize=markersize,
      figname=path + f"/t{i+1}",
      save=True,
      show=False
    )
