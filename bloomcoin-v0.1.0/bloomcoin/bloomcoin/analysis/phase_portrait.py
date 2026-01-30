"""
BloomCoin Phase Portrait Visualization

Visualize Kuramoto oscillator dynamics.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.animation import FuncAnimation
from typing import Optional, List, Tuple
import matplotlib.colors as mcolors


def plot_phase_portrait(
    phases: np.ndarray,
    r: Optional[float] = None,
    psi: Optional[float] = None,
    title: str = "Phase Portrait",
    ax: Optional[plt.Axes] = None,
    show_threshold: bool = True,
    show_distribution: bool = False,
    colormap: str = 'viridis'
) -> plt.Axes:
    """
    Plot oscillator phases on unit circle.

    Args:
        phases: Array of oscillator phases
        r: Order parameter (computed if None)
        psi: Mean phase (computed if None)
        title: Plot title
        ax: Matplotlib axes (creates if None)
        show_threshold: Draw z_c threshold circle
        show_distribution: Color by density
        colormap: Colormap for density visualization

    Returns:
        Matplotlib axes
    """
    from ..constants import Z_C

    # Compute order parameter if needed
    if r is None or psi is None:
        phases_complex = np.exp(1j * phases)
        mean_phase_complex = np.mean(phases_complex)
        r = np.abs(mean_phase_complex)
        psi = np.angle(mean_phase_complex)

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    # Unit circle
    theta = np.linspace(0, 2*np.pi, 100)
    ax.plot(np.cos(theta), np.sin(theta), 'k-', alpha=0.3, linewidth=1)

    # Threshold circle (z_c)
    if show_threshold:
        ax.plot(Z_C * np.cos(theta), Z_C * np.sin(theta),
                'g--', alpha=0.5, linewidth=2, label=f'z_c = {Z_C:.3f}')

    # Oscillators
    x = np.cos(phases)
    y = np.sin(phases)

    if show_distribution:
        # Color by local density
        from scipy.stats import gaussian_kde

        # Estimate density
        positions = np.vstack([x, y])
        if len(phases) > 1:
            try:
                kde = gaussian_kde(positions)
                density = kde(positions)
                # Normalize density for coloring
                density_norm = (density - density.min()) / (density.max() - density.min())
                colors = plt.cm.get_cmap(colormap)(density_norm)
                ax.scatter(x, y, c=colors, alpha=0.8, s=40, zorder=5)
            except:
                # Fallback to uniform color if KDE fails
                ax.scatter(x, y, c='blue', alpha=0.6, s=30, zorder=5)
        else:
            ax.scatter(x, y, c='blue', alpha=0.6, s=30, zorder=5)
    else:
        ax.scatter(x, y, c='blue', alpha=0.6, s=30, zorder=5)

    # Order parameter arrow
    arrow_scale = 0.9
    ax.arrow(0, 0, r*np.cos(psi)*arrow_scale, r*np.sin(psi)*arrow_scale,
             head_width=0.05, head_length=0.03,
             fc='red', ec='red', linewidth=2, zorder=10,
             label=f'r = {r:.4f}')

    # Center dot
    ax.scatter([0], [0], c='black', s=50, zorder=6)

    # Add phase distribution histogram on the side
    if show_distribution:
        # Create inset for histogram
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        axins = inset_axes(ax, width="30%", height="30%", loc='upper right')

        # Phase histogram
        n_bins = 36
        hist, bins = np.histogram(phases, bins=n_bins, range=(0, 2*np.pi))
        bin_centers = (bins[:-1] + bins[1:]) / 2

        # Polar histogram
        axins.bar(bin_centers, hist, width=bins[1]-bins[0], alpha=0.7, color='blue')
        axins.set_xlabel('Phase (rad)')
        axins.set_ylabel('Count')
        axins.set_xlim(0, 2*np.pi)
        axins.grid(True, alpha=0.3)

    # Formatting
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('Real')
    ax.set_ylabel('Imaginary')
    ax.set_title(f'{title}\nr = {r:.4f}, ψ = {psi:.4f}')
    ax.legend(loc='upper left')

    return ax


def plot_phase_evolution(
    state_history: List[Tuple[np.ndarray, float, float]],
    title: str = "Phase Evolution",
    figsize: Tuple[int, int] = (15, 5)
) -> plt.Figure:
    """
    Plot evolution of phases and order parameter over time.

    Args:
        state_history: List of (phases, r, psi) tuples
        title: Plot title
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    from ..constants import Z_C

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Extract data
    times = list(range(len(state_history)))
    r_values = [state[1] for state in state_history]
    psi_values = [state[2] for state in state_history]

    # Plot 1: Order parameter over time
    ax1 = axes[0]
    ax1.plot(times, r_values, 'b-', linewidth=2, label='r(t)')
    ax1.axhline(y=Z_C, color='g', linestyle='--', alpha=0.7, label=f'z_c = {Z_C:.3f}')
    ax1.fill_between(times, 0, r_values, alpha=0.3, color='blue')
    ax1.set_xlabel('Time step')
    ax1.set_ylabel('Order parameter r')
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_title('Coherence Evolution')

    # Plot 2: Mean phase over time
    ax2 = axes[1]
    ax2.plot(times, psi_values, 'r-', linewidth=2)
    ax2.set_xlabel('Time step')
    ax2.set_ylabel('Mean phase ψ (rad)')
    ax2.set_ylim(-np.pi, np.pi)
    ax2.grid(True, alpha=0.3)
    ax2.set_title('Mean Phase Evolution')

    # Plot 3: Final phase portrait
    ax3 = axes[2]
    final_phases, final_r, final_psi = state_history[-1]
    plot_phase_portrait(final_phases, final_r, final_psi,
                        title='Final State', ax=ax3, show_threshold=True)

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()

    return fig


def animate_kuramoto(
    state_history: List[Tuple[np.ndarray, float, float]],
    interval: int = 50,
    save_path: Optional[str] = None,
    show_trail: bool = False
) -> FuncAnimation:
    """
    Animate Kuramoto oscillator evolution.

    Args:
        state_history: List of (phases, r, psi) tuples
        interval: Milliseconds between frames
        save_path: Path to save animation (None = display)
        show_trail: Show trail of order parameter

    Returns:
        FuncAnimation object
    """
    from ..constants import Z_C

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Phase portrait
    ax1.set_xlim(-1.3, 1.3)
    ax1.set_ylim(-1.3, 1.3)
    ax1.set_aspect('equal')
    ax1.set_xlabel('Real')
    ax1.set_ylabel('Imaginary')
    ax1.grid(True, alpha=0.3)

    # Right: Order parameter over time
    r_values = [state[1] for state in state_history]
    ax2.set_xlim(0, len(state_history))
    ax2.set_ylim(0, 1)
    ax2.axhline(y=Z_C, color='g', linestyle='--', alpha=0.7, label=f'z_c = {Z_C:.3f}')
    ax2.set_xlabel('Time step')
    ax2.set_ylabel('Order parameter r')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Initialize plot elements
    scatter = ax1.scatter([], [], c='blue', alpha=0.6, s=30)
    arrow = None
    line, = ax2.plot([], [], 'b-', linewidth=2)
    point, = ax2.plot([], [], 'ro', markersize=8)

    # Trail for order parameter (optional)
    if show_trail:
        trail_x = []
        trail_y = []
        trail_line, = ax1.plot([], [], 'r-', alpha=0.3, linewidth=1)

    # Unit circle and threshold
    theta = np.linspace(0, 2*np.pi, 100)
    ax1.plot(np.cos(theta), np.sin(theta), 'k-', alpha=0.3)
    ax1.plot(Z_C * np.cos(theta), Z_C * np.sin(theta), 'g--', alpha=0.5)

    def init():
        scatter.set_offsets(np.empty((0, 2)))
        line.set_data([], [])
        point.set_data([], [])
        if show_trail:
            trail_line.set_data([], [])
        return scatter, line, point

    def update(frame):
        nonlocal arrow

        phases, r, psi = state_history[frame]

        # Update scatter (oscillators)
        x = np.cos(phases)
        y = np.sin(phases)
        scatter.set_offsets(np.column_stack([x, y]))

        # Update arrow (order parameter)
        if arrow:
            arrow.remove()
        arrow = ax1.arrow(0, 0, r*np.cos(psi)*0.9, r*np.sin(psi)*0.9,
                         head_width=0.05, head_length=0.03,
                         fc='red', ec='red', linewidth=2)

        # Update trail
        if show_trail and frame > 0:
            trail_x.append(r * np.cos(psi))
            trail_y.append(r * np.sin(psi))
            if len(trail_x) > 50:  # Limit trail length
                trail_x.pop(0)
                trail_y.pop(0)
            trail_line.set_data(trail_x, trail_y)

        # Update line (r over time)
        line.set_data(range(frame+1), r_values[:frame+1])
        point.set_data([frame], [r_values[frame]])

        # Update title
        ax1.set_title(f'Phase Portrait (t={frame})\nr = {r:.4f}')

        return scatter, arrow, line, point

    anim = FuncAnimation(fig, update, init_func=init,
                         frames=len(state_history), interval=interval,
                         blit=False, repeat=True)

    if save_path:
        # Save animation
        try:
            anim.save(save_path, writer='pillow', fps=1000/interval)
            print(f"Animation saved to {save_path}")
        except Exception as e:
            print(f"Failed to save animation: {e}")

    plt.suptitle('Kuramoto Oscillator Evolution')
    plt.tight_layout()

    return anim


def plot_coherence_heatmap(
    coupling_range: Optional[np.ndarray] = None,
    frequency_std_range: Optional[np.ndarray] = None,
    n_trials: int = 10,
    n_oscillators: int = 63,
    n_steps: int = 500,
    resolution: int = 20
) -> plt.Figure:
    """
    Create heatmap of final coherence vs coupling and frequency spread.

    Shows phase transition boundary where r crosses z_c.

    Args:
        coupling_range: Range of coupling values
        frequency_std_range: Range of frequency standard deviations
        n_trials: Number of trials per point
        n_oscillators: Number of oscillators
        n_steps: Integration steps per trial
        resolution: Grid resolution

    Returns:
        Matplotlib figure
    """
    from ..constants import Z_C, K

    if coupling_range is None:
        coupling_range = np.linspace(0, 2*K, resolution)

    if frequency_std_range is None:
        frequency_std_range = np.linspace(0, 2, resolution)

    print(f"Generating coherence heatmap ({resolution}x{resolution} grid)...")

    coherence_matrix = np.zeros((len(frequency_std_range), len(coupling_range)))

    for i, freq_std in enumerate(frequency_std_range):
        print(f"  Progress: {i+1}/{len(frequency_std_range)}", end='\r')
        for j, coupling in enumerate(coupling_range):
            r_finals = []

            for _ in range(n_trials):
                # Initialize random phases
                phases = np.random.uniform(0, 2*np.pi, n_oscillators)

                # Natural frequencies
                if freq_std > 0:
                    frequencies = np.random.normal(0, freq_std, n_oscillators)
                else:
                    frequencies = np.zeros(n_oscillators)

                # Simple Kuramoto evolution
                dt = 0.01
                for _ in range(n_steps):
                    # Compute mean field
                    phases_complex = np.exp(1j * phases)
                    mean_field = np.mean(phases_complex)
                    r_inst = np.abs(mean_field)
                    psi_inst = np.angle(mean_field)

                    # Update phases
                    phases += dt * (frequencies + coupling * r_inst * np.sin(psi_inst - phases))
                    phases = phases % (2*np.pi)

                # Final r
                phases_complex = np.exp(1j * phases)
                r_final = np.abs(np.mean(phases_complex))
                r_finals.append(r_final)

            coherence_matrix[i, j] = np.mean(r_finals)

    print("\nGenerating plot...")

    fig, ax = plt.subplots(figsize=(10, 8))

    # Create heatmap
    im = ax.imshow(coherence_matrix, aspect='auto', origin='lower',
                   extent=[coupling_range[0], coupling_range[-1],
                           frequency_std_range[0], frequency_std_range[-1]],
                   cmap='viridis', vmin=0, vmax=1)

    # Draw z_c contour (phase boundary)
    contour = ax.contour(coupling_range, frequency_std_range, coherence_matrix,
                         levels=[Z_C], colors='red', linewidths=2)
    ax.clabel(contour, inline=True, fontsize=10, fmt=f'r = {Z_C:.3f}')

    # Add additional contours
    ax.contour(coupling_range, frequency_std_range, coherence_matrix,
               levels=[0.2, 0.4, 0.6, 0.8], colors='white', linewidths=0.5, alpha=0.5)

    # Mark theoretical coupling K
    ax.axvline(x=K, color='yellow', linestyle='--', alpha=0.7, label=f'K = {K:.3f}')

    ax.set_xlabel('Coupling strength K', fontsize=12)
    ax.set_ylabel('Frequency spread σ', fontsize=12)
    ax.set_title('Kuramoto Phase Diagram\nRed contour: r = z_c (phase boundary)', fontsize=14)
    ax.legend()

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, label='Order parameter r')
    cbar.ax.axhline(y=Z_C, color='red', linewidth=2)

    plt.tight_layout()

    return fig


def plot_synchronization_path(
    phases_history: List[np.ndarray],
    title: str = "Synchronization Path"
) -> plt.Figure:
    """
    Visualize the path to synchronization in phase space.

    Args:
        phases_history: List of phase arrays over time
        title: Plot title

    Returns:
        Matplotlib figure
    """
    from ..constants import Z_C

    n_snapshots = min(6, len(phases_history))
    indices = np.linspace(0, len(phases_history)-1, n_snapshots, dtype=int)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i, idx in enumerate(indices):
        phases = phases_history[idx]

        # Compute order parameter
        phases_complex = np.exp(1j * phases)
        mean_complex = np.mean(phases_complex)
        r = np.abs(mean_complex)
        psi = np.angle(mean_complex)

        # Plot on subplot
        ax = axes[i]
        plot_phase_portrait(
            phases, r, psi,
            title=f't = {idx} (r = {r:.3f})',
            ax=ax,
            show_threshold=True,
            show_distribution=False
        )

        # Color based on synchronization state
        if r > Z_C:
            ax.patch.set_facecolor('#e6ffe6')  # Light green
        elif r > Z_C - 0.1:
            ax.patch.set_facecolor('#ffffe6')  # Light yellow
        else:
            ax.patch.set_facecolor('#ffe6e6')  # Light red

    fig.suptitle(title, fontsize=16)
    plt.tight_layout()

    return fig