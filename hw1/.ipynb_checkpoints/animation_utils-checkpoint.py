import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def animate_wavefunction(psi_matrix, x, time, V_0, x_V, h, x_lims, filename="wavefunction.gif"):
    """
    Creates a GIF animation of the wavefunction evolution with an overlay of the potential barrier.
    
    This function generates an animation that visualizes the time evolution of a quantum wavefunction.
    It displays the probability density |ψ(x)|² on the top subplot and the real and imaginary parts
    of ψ(x) on the bottom subplot. Additionally, it overlays a representation of the potential barrier
    on both subplots using a secondary y-axis.
    
    Parameters:
        psi_matrix (np.ndarray):
            A 2D array where each row corresponds to the wavefunction ψ(x) at a specific time.
        x (np.ndarray):
            An array of spatial coordinates.
        time (array-like):
            A sequence of time values corresponding to each row in psi_matrix.
        V_0 (float):
            The height of the potential barrier.
        x_V (float):
            The starting x-coordinate of the potential barrier.
        h (float):
            The width of the potential barrier.
        x_lims (tuple or list):
            The limits for the x-axis in the plots, e.g., (x_min, x_max).
        filename (str, optional):
            The name of the output GIF file where the animation will be saved. Defaults to "wavefunction.gif".
    
    Returns:
        None. The animation is saved to the specified filename and the figure is closed.
    """
    fig, axs = plt.subplots(2, 1, figsize=(8, 10))

    line_abs, = axs[0].plot([], [], 'b-', lw=2)
    line_re, = axs[1].plot([], [], 'r-', lw=2, label=r"Re($\psi$)")
    line_im, = axs[1].plot([], [], 'g-', lw=2, label=r"Im($\psi$)")

    for ax in axs:
        ax.set_xlim(x_lims[0], x_lims[1])

    abs_max = np.max(np.abs(psi_matrix)**2)
    axs[0].set_ylim(0, 1)
    axs[0].set_title(r"$|\psi|^2$")
    axs[0].set_xlabel("x")
    axs[0].set_ylabel(r"$|\psi|^2$")

    re_vals = np.real(psi_matrix)
    im_vals = np.imag(psi_matrix)
    y_min = min(np.min(re_vals), np.min(im_vals))
    y_max = max(np.max(re_vals), np.max(im_vals))
    margin = 0.1 * (y_max - y_min) if (y_max - y_min) != 0 else 1
    axs[1].set_ylim(-1, 1)
    axs[1].set_title(r"Re($\psi$) and Im($\psi$)")
    axs[1].set_xlabel("x")
    axs[1].set_ylabel("Values")
    axs[1].legend()

    # Create a secondary axis to display the potential barrier
    ax2_0 = axs[0].twinx()
    ax2_1 = axs[1].twinx()
    ax2_0.set_ylabel("Energy", color="purple")
    ax2_1.set_ylabel("Energy", color="purple")
    if V_0 > 0:
        ax2_0.set_ylim(0, V_0 * 1.1)
        ax2_1.set_ylim(0, V_0 * 1.1)
    else:
        ax2_0.set_ylim(0, 1)
        ax2_1.set_ylim(0, 1)

    V_x = [x_V, x_V, x_V + h, x_V + h]
    V_y = [0, V_0, V_0, 0]
    barrier_0, = ax2_0.plot(V_x, V_y, 'm-', lw=3)
    barrier_1, = ax2_1.plot(V_x, V_y, 'm-', lw=3)

    def init():
        line_abs.set_data([], [])
        line_re.set_data([], [])
        line_im.set_data([], [])
        return line_abs, line_re, line_im, barrier_0, barrier_1

    def update(frame):
        psi = psi_matrix[frame]
        line_abs.set_data(x, np.abs(psi)**2)
        line_re.set_data(x, np.real(psi))
        line_im.set_data(x, np.imag(psi))
        fig.suptitle(f"t = {time[frame]:.4f}")
        return line_abs, line_re, line_im, barrier_0, barrier_1

    ani = animation.FuncAnimation(fig, update, frames=len(time), init_func=init,
                                  blit=True, interval=50)

    ani.save(filename, writer="pillow", fps=15)
    plt.close(fig)