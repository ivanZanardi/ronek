import matplotlib
import matplotlib.pyplot as plt

from IPython.display import HTML
from matplotlib.animation import FuncAnimation

COLORS = matplotlib.rcParams["axes.prop_cycle"].by_key()["color"]


# Animation
# =====================================
# Initialize lines for levels distribution
def _init_lines(
  y,
  ax,
  markersize
):
  # Set up axes
  ax.set_xlabel(r"$\epsilon_i$ [eV]")
  ax.set_ylabel(r"$n_i\//\/g_i$ [m$^{-3}$]")
  # Initialize lines
  style = dict(
    linestyle="",
    marker="o",
    fillstyle="full"
  )
  i = 0
  colors = []
  for yi in y:
    if ("FOM" in yi[0].upper()):
      colors.append("k")
    else:
      colors.append(COLORS[i])
      i += 1
  lines = []
  for c in colors:
    lines.append(ax.semilogy([], [], c=c, markersize=markersize, **style)[0])
  # Add legend
  ax.legend(
    [ax.semilogy([], [], c=c, markersize=6, **style)[0] for c in colors],
    labels=[yi[0] for yi in y],
    loc="lower left"
  )
  return ax, lines

# Create animation
def _create_animation(
  t,
  x,
  y,
  frames,
  markersize
):
  # Initialize a figure in which the graphs will be plotted
  fig, ax = plt.subplots()
  # Initialize levels distribution lines objects
  ax, lines = _init_lines(y, ax, markersize)
  # Initialize text in ax
  txt = ax.text(0.7, 0.92, "", transform=ax.transAxes, fontsize=25)
  # Tight layout
  plt.tight_layout()

  def _animate(frame):
    i = int(frame/frames*len(t))
    # Write time instant
    txt.set_text(r"$t$ = %.1e s" % t[i])
    # Loop over models
    for (j, yj) in enumerate(y):
      lines[j].set_data(x, yj[1][i])
    # Rescale axis limits
    ax.relim()
    ax.autoscale_view(tight=True)
    return lines

  # Get animation
  return FuncAnimation(
    fig,
    _animate,
    frames=frames,
    blit=True
  )

def animate(
  t,
  x,
  y,
  markersize=6,
  frames=10,
  fps=10,
  filename="./lev_dist.gif",
  dpi=600,
  save=True,
  show=False
):
  # Create animation
  anim = _create_animation(t, x, y, frames, markersize)
  # Save animation
  if save:
    anim.save(filename, fps=fps, dpi=dpi)
  # Display animation
  if show:
    HTML(anim.to_jshtml())
