import os
import numpy as np
import scipy as sp
import matplotlib
import matplotlib.pyplot as plt

from .. import const
from .. import utils

COLORS = matplotlib.rcParams["axes.prop_cycle"].by_key()["color"]


# Plotting
# =====================================
# Cumulative energy
def plot_cum_energy(
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
def plot_evolution(
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

# Moments
def plot_mom_evolution(
  path,
  t,
  n_m,
  molecule,
  max_mom=2
):
  path = path + "/moments/"
  os.makedirs(path, exist_ok=True)
  # Compute moments
  moms = [{k: molecule.compute_mom(nk, m=0) for (k, nk) in n_m.items()}]
  for m in range(1,max_mom):
    moms.append(
      {k: molecule.compute_mom(nk, m=1)/moms[0][k] for (k, nk) in n_m.items()}
    )
  # Plot moments
  for m in range(max_mom):
    if (m == 0):
      yscale = "log"
      label_sol = r"$n$ [m$^{-3}$]"
    else:
      yscale = "linear"
      if (m == 1):
        label_sol = r"$e_{\text{int}}$ [eV]"
      else:
        label_sol = fr"$\gamma_{m}$ [eV$^{m}$]"
    # > Moment
    plot_evolution(
      x=t,
      y=moms[m],
      labels=[r"$t$ [s]", label_sol],
      scales=["log", yscale],
      figname=path + f"/m{m}",
      save=True,
      show=False
    )
    # > Moment error
    for (k, momk) in moms[m].items():
      if ("FOM" in k.upper()):
        mom_fom = momk
        break
    moms_err = {}
    for (k, momk) in moms[m].items():
      if ("FOM" not in k.upper()):
        moms_err[k] = utils.absolute_percentage_error(momk, mom_fom)
    plot_evolution(
      x=t,
      y=moms_err,
      labels=[r"$t$ [s]", r"Error [\%]"],
      scales=['log', 'linear'],
      figname=path + f"/m{m}_err",
      save=True,
      show=False
    )

# 2D distribution
def plot_dist_2d(
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

def plot_multi_dist_2d(
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
    plot_dist_2d(
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
