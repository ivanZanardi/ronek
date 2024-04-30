import numpy as np

from matplotlib import ticker
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation


# Plotting
# =====================================
style = dict(
  markeredgewidth=1,
  lw=1.0,
  linestyle="",
  markersize=1.0,
  marker="x"
)

# 2D distribution
# -------------------------------------
def plot_2D(
  x,
  y_true,
  y_pred=None,
  labels=[r"$\epsilon_i$ [eV]", r"$n_i/g_i$ [m$^{-3}$]"],
  scales=["linear", "log"],
  filename=None,
  save=False,
  show=False
):
  # Initialize figures
  fig = plt.figure()
  ax = fig.add_subplot()
  # x axis
  xlim = [np.amin(x), np.amax(x)]
  ax.set_xlim(xlim)
  ax.set_xlabel(labels[0])
  ax.set_yscale(scales[0])
  # y axis
  ax.set_ylabel(labels[1])
  ax.set_yscale(scales[1])
  # Plotting
  ax.plot(x, y_true, c="k", **style)
  if (y_pred is not None):
    ax.plot(x, y_pred, c="r", **style)
  if save:
    plt.savefig(filename)
  if show:
    plt.show()
  else:
    plt.close()

# 3D distribution
# -------------------------------------
def plot_3D(
  x,
  y,
  z_true,
  z_pred=None,
  z_label=r"$n_i/g_i$ [m$^{-3}$]",
  z_scale="log",
  diss_plane=False,
  diss_energy=0.0,
  elev=30.0,
  axim=0.0,
  filename=None,
  save=False,
  show=False
):
  # Initialize figures
  fig = plt.figure()
  ax = fig.add_subplot(projection="3d")
  # x axis
  xlim = [np.amin(x), np.amax(x)]
  ax.set_xlim(xlim)
  ax.set_xlabel(r"$\epsilon_v$ [eV]")
  # y axis
  ylim = [np.amin(y), np.amax(y)]
  ax.set_ylim(ylim)
  ax.set_ylabel(r"$\epsilon_r$ [eV]")
  # z axis
  ax.set_zlabel(z_label)
  if (z_scale == "log"):
    ax.zaxis.set_major_formatter(
      ticker.FuncFormatter(log_tick_formatter)
    )
    ax.zaxis.set_major_locator(
      ticker.MaxNLocator(integer=True)
    )
    z_true = np.log10(z_true)
    if (z_pred is not None):
      z_pred = np.log10(z_pred)
  # Plotting
  ax.plot3D(x, y, z_true, c="k", **style)
  if (z_pred is not None):
    ax.plot3D(x, y, z_pred, c="r", **style)
  plt.tight_layout()
  # Dissociation limit plane
  if diss_plane:
    xs = np.array(xlim)
    zs = np.array([np.amin(z_true), np.amax(z_true)])
    X, Z = np.meshgrid(xs, zs)
    Y = diss_energy - X
    ax.plot_surface(X, Y, Z, color="b", alpha=0.1)
  if save:
    plt.savefig(filename)
  if show:
    plt.show()
  else:
    plt.close()

# Cumulative energy
def plot_cumenergy(
  y,
  filename=None,
  save=False,
  show=True
):
  # Initialize figure
  fig = plt.figure()
  ax = fig.add_subplot()
  # x axis
  x = np.arange(1,len(y)+1)
  ax.set_xlabel("$k$-th basis")
  # y axis
  ax.set_ylabel("Cumulative energy")
  # Plotting
  ax.plot(
    x,
    y,
    markeredgewidth=1,
    lw=1.0,
    markersize=5.0,
    marker="o"
  )
  if save:
    plt.savefig(filename)
  if show:
    plt.show()
  else:
    plt.close()

# 3D Animation
# -------------------------------------
# Log-scale ticker
def log_tick_formatter(val, pos=None):
  return f"$10^{{{int(val)}}}$"

# Initialize lines for levels distribution
def init_3D_lines(
  ax,
  axim=0.0
):
  # Set up axes
  ax.set_xlabel(r"$\epsilon_v$ [eV]")
  ax.set_ylabel(r"$\epsilon_r$ [eV]")
  ax.set_zlabel(r"$n_i/g_i$ [m$^{-3}$]")
  ax.zaxis.set_major_formatter(
    ticker.FuncFormatter(log_tick_formatter)
  )
  ax.zaxis.set_major_locator(
    ticker.MaxNLocator(integer=True)
  )
  # Initialize lines.Line2D objects
  lines = [ax.plot3D([],[],[],c=c,**style)[0] for c in ("k","r")]
  ax.view_init(elev=30, azim=axim)
  return ax, lines

# Create animation
def get_3D_animation(
  t_vec,
  n_ov_g,
  frames,
  axim=0.,
  lev_dtb=None
):
  # Initialize a figure in which the graphs will be plotted
  fig = plt.figure()
  ax = fig.add_subplot(projection="3d")
  # Initialize levels distribution lines objects
  ax, lines = init_3D_lines(ax, axim=axim)
  xlim = [np.amin(lev_dtb["EVib"]), np.amax(lev_dtb["EVib"])]
  ylim = [np.amin(lev_dtb["ERot"]), np.amax(lev_dtb["ERot"])]
  t_text = ax.text2D(0.05, 0.95, "", transform=ax.transAxes)

  def animate(frame):
    print(f"Frame {frame+1}/{frames}", end = "\r")
    idx = int(frame/frames*len(t_vec))
    t_text.set_text(r"$t$ = %.2e s" % t_vec[idx])
    # Plot distribution
    for l in lines:
      l.set_xdata(lev_dtb["EVib"])
      l.set_ydata(lev_dtb["ERot"])
    true = np.log10(n_ov_g[0][idx])
    lines[0].set_3d_properties(true)
    pred = np.log10(n_ov_g[1][idx])
    lines[1].set_3d_properties(pred)
    # Rescale axis limits
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim([np.amin(true), np.amax(true)])
    return lines

  # Get animation
  return FuncAnimation(
    fig,
    animate,
    frames=frames,
    blit=True
  )

# 2D Animation
# -------------------------------------
# Initialize lines for levels distribution
def init_2D_lines(
  ax
):
  # Set up axes
  ax.set_xlabel(r"$\epsilon_i$ [eV]")
  ax.set_ylabel(r"$n_i/g_i$ [m$^{-3}$]")
  ax.set_yscale("log")
  # Initialize lines.Line2D objects
  lines = [ax.plot([],[],c=c,**style)[0] for c in ("k","r")]
  return ax, lines

# Create animation
def get_2D_animation(
  t_vec,
  n_ov_g,
  frames,
  lev_dtb=None
):
  # Initialize a figure in which the graphs will be plotted
  fig = plt.figure()
  ax = fig.add_subplot()
  # Initialize levels distribution lines objects
  ax, lines = init_2D_lines(ax)
  xlim = [np.amin(lev_dtb["E"]), np.amax(lev_dtb["E"])]
  t_text = ax.text(0.75, 0.9, "", transform=ax.transAxes)

  def animate(frame):
    print(f"Frame {frame+1}/{frames}", end="\r")
    idx = int(frame/frames*len(t_vec))
    t_text.set_text(r"$t$ = %.2e s" % t_vec[idx])
    # Plot distribution
    for l in lines:
      l.set_xdata(lev_dtb["E"])
    for i in range(2):
      lines[i].set_ydata(n_ov_g[i,idx])
    # Rescale axis limits
    ax.set_xlim(xlim)
    ax.set_ylim([np.amin(n_ov_g[0,idx]), np.amax(n_ov_g[0,idx])])
    return lines

  # Get animation
  return FuncAnimation(
    fig,
    animate,
    frames=frames,
    blit=True
  )
