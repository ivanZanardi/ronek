import matplotlib as mpl

from IPython.display import HTML
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

# Get colors list
_COLORS = mpl.rcParams['axes.prop_cycle'].by_key()['color']*10


# Animation
# =====================================
# Initialize lines for levels distribution
def _init_lines(ax, varnames):
  # Set up axes
  ax.set_xlabel(r'$\epsilon_i$ [eV]')
  ax.set_ylabel(r'$n_i\//\/g_i$ [m$^{-3}$]')
  # Define plot style
  style = dict(
    lw=1,
    marker='x',
    fillstyle='full',
    markersize=0.3,
    linestyle=''
  )
  # Add legend
  ax.legend(
    [ax.plot([], [], c=_COLORS[i])[0] for i in range(len(varnames))],
    varnames,
    loc='lower left'
  )
  # Initialize lines.Line2D objects
  lines = [
    ax.semilogy([], [], c=_COLORS[i], **style)[0] \
      for i in range(len(varnames))
  ]
  return ax, lines

# Create animation
def _create_animation(t, x, y, frames):
  # Initialize a figure in which the graphs will be plotted
  fig, ax = plt.subplots()
  # Initialize levels distribution lines objects
  ax, lines = _init_lines(ax, varnames=list(y.keys()))
  # Initialize text in ax
  txt = ax.text(0.77, 0.92, '', transform=ax.transAxes, fontsize=10)

  def _animate(frame):
    idx = int(frame/frames*len(t))
    # Write time instant
    txt.set_text(r'$t$ = %.1e s' % t[idx])
    # Loop over Timescales
    for (i, yi) in enumerate(y.values()):
      lines[i].set_data(x, yi[idx])
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
  frames=10,
  fps=10,
  filename="./lev_dist.gif",
  dpi=600,
  save=True,
  show=False
):
  # Create animation
  anim = _create_animation(t, x, y, frames)
  # Save animation
  if save:
    anim.save(filename, fps=fps, dpi=dpi)
  # Display animation
  if show:
    HTML(anim.to_jshtml())
