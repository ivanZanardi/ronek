import os
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

# Use custom style
# See: https://matplotlib.org/1.5.3/users/style_sheets.html
# style_name = 'paper_1column'
# mpl.style.use(
#     os.environ['WORKSPACE_PATH'] \
#         + '/Research/styles/matplotlib/' \
#         + style_name + '.mplstyle'
# )
# Get colors list
COLORS = mpl.rcParams['axes.prop_cycle'].by_key()['color']*10


# Initialize lines for levels distribution
def init_lines(
    ax,
    var_names
):
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
        [ax.plot([], [], c=COLORS[i])[0] for i in range(len(var_names))],
        var_names,
        loc='lower left'
    )
    # Initialize lines.Line2D objects
    lines = [
        ax.semilogy([], [], c=COLORS[i], **style)[0] \
            for i in range(len(var_names))
    ]
    return ax, lines

# Create animation
def create_animation(t, x, y, frames):
    # Initialize a figure in which the graphs will be plotted
    fig, ax = plt.subplots()
    # Initialize levels distribution lines objects
    ax, lines = init_lines(ax, var_names=list(y.keys()))
    # Initialize text in ax
    txt = ax.text(0.77, 0.92, '', transform=ax.transAxes, fontsize=10)

    def animate(frame):
        idx = int(frame/frames*len(t))
        # Write time instant
        txt.set_text(r'$t$ = %.1e s' % t[idx])
        # Loop over Timescales
        for i, Xi in enumerate(y.values()):
            lines[i].set_data(x, Xi[idx])
        # Rescale axis limits
        ax.relim()
        ax.autoscale_view(tight=True)
        return lines
        
    # Get animation
    return FuncAnimation(
        fig,
        animate,
        frames=frames,
        blit=True
    )

# Create animation
def animate(
    t,
    x,
    y,
    frames=10,
    fps=10,
    filename="./lev_dist.mp4",
    save=True,
    show=False
):
    # Create animation
    anim = create_animation(t, x, y, frames)
    # Save animation
    if save:
        anim.save(filename, fps=fps, dpi=900)
    # Display animation
    if show:
        HTML(anim.to_jshtml())