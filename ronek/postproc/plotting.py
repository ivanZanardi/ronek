import numpy as np

from matplotlib import pyplot as plt


# Plotting
# =====================================
# 2D distribution
def dist_2d(
  x,
  y,
  y_pred=None,
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
    marker="o",
    markersize=markersize
  )
  ax.plot(x, y, c="k", **style)
  if (y_pred is not None):
    ax.plot(x, y_pred, c="r", **style)
    lines = [
      plt.plot([], [], c="k", **style)[0],
      plt.plot([], [], c="r", **style)[0]
    ]
    ax.legend(lines, ["FOM", "ROM"])
  # Tight layout
  plt.tight_layout()
  if save:
    plt.savefig(figname)
  if show:
    plt.show()
  plt.close()

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
  y_pred=None,
  labels=[r"$t$ [s]", r"$n$ [m$^{-3}$]"],
  scales=['log', 'linear'],
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
  ax.plot(x, y, "-", c="k")
  if (y_pred is not None):
    ax.plot(x, y_pred, "--", c="r")
    lines = [
      plt.plot([], [], "-", c="k")[0],
      plt.plot([], [], "--", c="r")[0]
    ]
    ax.legend(lines, ["FOM", "ROM"])
  # Tight layout
  plt.tight_layout()
  if save:
    plt.savefig(figname)
  if show:
    plt.show()
  plt.close()
