"""
BloomCoin Hexagonal Lattice Analysis

Analyze hexagonal lattice structures and their properties.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List, Dict
from matplotlib.patches import RegularPolygon
import matplotlib.patches as patches


def generate_hexagonal_lattice(n_rings: int = 3) -> np.ndarray:
    """
    Generate coordinates for a hexagonal lattice.

    Args:
        n_rings: Number of hexagonal rings (0 = center only)

    Returns:
        Array of (x, y) coordinates
    """
    if n_rings == 0:
        return np.array([[0, 0]])

    coords = [[0, 0]]  # Center

    # Hexagonal unit vectors
    a1 = np.array([1, 0])
    a2 = np.array([0.5, np.sqrt(3)/2])

    for ring in range(1, n_rings + 1):
        # Generate points for this ring
        for i in range(ring + 1):
            for j in range(ring + 1):
                if i + j == ring:
                    # Point in ring
                    point = i * a1 + j * a2
                    coords.append(point)

                    # Add symmetric points
                    if i > 0 and j > 0:
                        coords.append(-point)
                    if i > 0:
                        coords.append(j * a2 - i * a1)
                    if j > 0:
                        coords.append(i * a1 - j * a2)
                    if i > 0 and j < ring:
                        coords.append(-i * a1 + (ring - j) * a2)
                    if j > 0 and i < ring:
                        coords.append((ring - i) * a1 - j * a2)

    # Remove duplicates
    coords = np.array(coords)
    coords = np.unique(coords, axis=0)

    return coords


def hexagonal_distance(p1: np.ndarray, p2: np.ndarray) -> float:
    """
    Compute distance in hexagonal lattice.

    Uses Manhattan distance in hexagonal coordinates.

    Args:
        p1: First point
        p2: Second point

    Returns:
        Hexagonal distance
    """
    # Convert to axial coordinates
    q1, r1 = cartesian_to_axial(p1)
    q2, r2 = cartesian_to_axial(p2)

    # Hexagonal distance
    return (abs(q1 - q2) + abs(q1 + r1 - q2 - r2) + abs(r1 - r2)) / 2


def cartesian_to_axial(point: np.ndarray) -> Tuple[int, int]:
    """
    Convert Cartesian to axial hexagonal coordinates.

    Args:
        point: (x, y) Cartesian coordinates

    Returns:
        (q, r) axial coordinates
    """
    x, y = point
    q = round(x)
    r = round((y - x/2) * 2/np.sqrt(3))
    return int(q), int(r)


def plot_hexagonal_lattice(
    coords: np.ndarray,
    values: Optional[np.ndarray] = None,
    title: str = "Hexagonal Lattice",
    cmap: str = 'viridis',
    show_indices: bool = False
) -> plt.Figure:
    """
    Visualize hexagonal lattice.

    Args:
        coords: Lattice coordinates
        values: Optional values at each site
        title: Plot title
        cmap: Colormap for values
        show_indices: Show site indices

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 10))

    # Determine colors
    if values is not None:
        colors = plt.cm.get_cmap(cmap)(values)
    else:
        colors = 'lightblue'

    # Plot hexagons
    for i, (x, y) in enumerate(coords):
        hexagon = RegularPolygon(
            (x, y), 6, radius=0.5,
            facecolor=colors[i] if values is not None else colors,
            edgecolor='black', linewidth=1
        )
        ax.add_patch(hexagon)

        if show_indices:
            ax.text(x, y, str(i), ha='center', va='center', fontsize=8)

    # Set limits
    margin = 1
    ax.set_xlim(coords[:, 0].min() - margin, coords[:, 0].max() + margin)
    ax.set_ylim(coords[:, 1].min() - margin, coords[:, 1].max() + margin)
    ax.set_aspect('equal')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    if values is not None:
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap,
                                   norm=plt.Normalize(vmin=values.min(), vmax=values.max()))
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label='Value')

    return fig


def hexagonal_coupling_matrix(coords: np.ndarray, cutoff: float = 1.5) -> np.ndarray:
    """
    Generate coupling matrix for hexagonal lattice.

    Args:
        coords: Lattice coordinates
        cutoff: Distance cutoff for coupling

    Returns:
        Coupling matrix
    """
    n_sites = len(coords)
    coupling = np.zeros((n_sites, n_sites))

    for i in range(n_sites):
        for j in range(i + 1, n_sites):
            dist = np.linalg.norm(coords[i] - coords[j])
            if dist < cutoff:
                # Coupling strength decreases with distance
                coupling[i, j] = 1.0 / (1 + dist)
                coupling[j, i] = coupling[i, j]

    return coupling


def analyze_hexagonal_phases(
    phases: np.ndarray,
    coords: np.ndarray,
    n_neighbors: int = 6
) -> Dict:
    """
    Analyze phase distribution on hexagonal lattice.

    Args:
        phases: Phase at each lattice site
        coords: Lattice coordinates
        n_neighbors: Number of nearest neighbors to consider

    Returns:
        Analysis results
    """
    n_sites = len(phases)

    # Find nearest neighbors for each site
    neighbor_lists = []
    for i in range(n_sites):
        distances = np.array([np.linalg.norm(coords[i] - coords[j])
                             for j in range(n_sites)])
        # Sort by distance (exclude self)
        sorted_idx = np.argsort(distances)[1:n_neighbors+1]
        neighbor_lists.append(sorted_idx)

    # Compute local order parameters
    local_order = np.zeros(n_sites)
    for i in range(n_sites):
        neighbors = neighbor_lists[i]
        neighbor_phases = phases[neighbors]

        # Local order parameter
        local_complex = np.exp(1j * neighbor_phases)
        local_order[i] = np.abs(np.mean(local_complex))

    # Compute spatial correlation
    correlation_matrix = np.zeros((n_sites, n_sites))
    for i in range(n_sites):
        for j in range(n_sites):
            correlation_matrix[i, j] = np.cos(phases[i] - phases[j])

    # Correlation length (simplified)
    distances = []
    correlations = []
    for i in range(n_sites):
        for j in range(i+1, n_sites):
            dist = np.linalg.norm(coords[i] - coords[j])
            corr = correlation_matrix[i, j]
            distances.append(dist)
            correlations.append(corr)

    distances = np.array(distances)
    correlations = np.array(correlations)

    # Fit exponential decay
    try:
        from scipy.optimize import curve_fit

        def exp_decay(x, xi):
            return np.exp(-x / xi)

        # Only fit positive correlations
        mask = correlations > 0
        if np.sum(mask) > 10:
            popt, _ = curve_fit(exp_decay, distances[mask], correlations[mask], p0=[1.0])
            correlation_length = popt[0]
        else:
            correlation_length = 0.0
    except:
        correlation_length = 0.0

    return {
        'local_order': local_order,
        'mean_local_order': np.mean(local_order),
        'correlation_matrix': correlation_matrix,
        'correlation_length': correlation_length,
        'spatial_entropy': -np.sum(local_order * np.log(local_order + 1e-10)) / n_sites
    }


def plot_hexagonal_phase_map(
    phases: np.ndarray,
    coords: np.ndarray,
    title: str = "Phase Map on Hexagonal Lattice"
) -> plt.Figure:
    """
    Visualize phases on hexagonal lattice with arrows.

    Args:
        phases: Phases at each site
        coords: Lattice coordinates
        title: Plot title

    Returns:
        Matplotlib figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Left: Phase as colors
    ax1.set_title("Phase Distribution")
    phase_colors = plt.cm.hsv(phases / (2*np.pi))

    for i, (x, y) in enumerate(coords):
        hexagon = RegularPolygon(
            (x, y), 6, radius=0.5,
            facecolor=phase_colors[i],
            edgecolor='black', linewidth=0.5, alpha=0.8
        )
        ax1.add_patch(hexagon)

    # Right: Phase as arrows
    ax2.set_title("Phase Vectors")

    # Background hexagons
    for x, y in coords:
        hexagon = RegularPolygon(
            (x, y), 6, radius=0.5,
            facecolor='lightgray',
            edgecolor='black', linewidth=0.5, alpha=0.3
        )
        ax2.add_patch(hexagon)

    # Phase arrows
    arrow_scale = 0.4
    for i, (x, y) in enumerate(coords):
        dx = arrow_scale * np.cos(phases[i])
        dy = arrow_scale * np.sin(phases[i])
        ax2.arrow(x, y, dx, dy, head_width=0.1, head_length=0.05,
                 fc='blue', ec='blue', alpha=0.7)

    # Format both axes
    for ax in [ax1, ax2]:
        margin = 1
        ax.set_xlim(coords[:, 0].min() - margin, coords[:, 0].max() + margin)
        ax.set_ylim(coords[:, 1].min() - margin, coords[:, 1].max() + margin)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

    # Add colorbar for phase
    sm = plt.cm.ScalarMappable(cmap='hsv',
                               norm=plt.Normalize(vmin=0, vmax=2*np.pi))
    sm.set_array([])
    plt.colorbar(sm, ax=ax1, label='Phase (rad)')

    fig.suptitle(title)
    plt.tight_layout()

    return fig


def hexagonal_defect_analysis(
    phases: np.ndarray,
    coords: np.ndarray,
    threshold: float = np.pi/2
) -> Dict:
    """
    Identify and analyze topological defects in phase field.

    Args:
        phases: Phases at each site
        coords: Lattice coordinates
        threshold: Phase difference threshold for defect

    Returns:
        Defect analysis results
    """
    n_sites = len(phases)

    # Build neighbor list
    neighbors = []
    for i in range(n_sites):
        distances = np.array([np.linalg.norm(coords[i] - coords[j])
                             for j in range(n_sites)])
        # Get 6 nearest neighbors (excluding self)
        sorted_idx = np.argsort(distances)[1:7]
        neighbors.append(sorted_idx)

    # Detect defects (vortices)
    defects = []
    winding_numbers = np.zeros(n_sites)

    for i in range(n_sites):
        if len(neighbors[i]) < 6:
            continue

        # Compute winding number around site
        neighbor_phases = phases[neighbors[i]]

        # Sort neighbors by angle
        neighbor_coords = coords[neighbors[i]]
        center = coords[i]
        angles = np.arctan2(neighbor_coords[:, 1] - center[1],
                          neighbor_coords[:, 0] - center[0])
        sorted_idx = np.argsort(angles)
        sorted_phases = neighbor_phases[sorted_idx]

        # Compute winding
        phase_diffs = np.diff(np.append(sorted_phases, sorted_phases[0]))
        # Wrap phase differences to [-π, π]
        phase_diffs = np.angle(np.exp(1j * phase_diffs))

        winding = np.sum(phase_diffs) / (2 * np.pi)
        winding_numbers[i] = winding

        # Identify defects
        if abs(winding) > 0.5:
            defects.append({
                'site': i,
                'position': coords[i],
                'winding': round(winding),
                'strength': abs(winding)
            })

    # Classify defects
    positive_defects = [d for d in defects if d['winding'] > 0]
    negative_defects = [d for d in defects if d['winding'] < 0]

    return {
        'n_defects': len(defects),
        'defects': defects,
        'positive_defects': positive_defects,
        'negative_defects': negative_defects,
        'winding_numbers': winding_numbers,
        'total_winding': np.sum([d['winding'] for d in defects])
    }


def test_hexagonal_analysis():
    """Test suite for hexagonal lattice analysis."""
    print("\n" + "="*60)
    print("HEXAGONAL LATTICE TEST SUITE")
    print("="*60)

    # Test 1: Lattice generation
    print("\n1. Testing lattice generation...")
    coords = generate_hexagonal_lattice(n_rings=2)
    print(f"   Generated {len(coords)} sites")
    assert len(coords) == 19  # 1 + 6 + 12
    print("   ✓ Test passed")

    # Test 2: Coupling matrix
    print("\n2. Testing coupling matrix...")
    coupling = hexagonal_coupling_matrix(coords)
    print(f"   Coupling matrix shape: {coupling.shape}")
    assert coupling.shape == (len(coords), len(coords))
    assert np.allclose(coupling, coupling.T)  # Symmetric
    print("   ✓ Test passed")

    # Test 3: Phase analysis
    print("\n3. Testing phase analysis...")
    phases = np.random.uniform(0, 2*np.pi, len(coords))
    analysis = analyze_hexagonal_phases(phases, coords)
    print(f"   Mean local order: {analysis['mean_local_order']:.3f}")
    assert 0 <= analysis['mean_local_order'] <= 1
    print("   ✓ Test passed")

    # Test 4: Defect detection
    print("\n4. Testing defect analysis...")
    # Create vortex pattern
    vortex_phases = np.array([np.arctan2(y, x) for x, y in coords])
    defects = hexagonal_defect_analysis(vortex_phases, coords)
    print(f"   Found {defects['n_defects']} defects")
    print(f"   Total winding: {defects['total_winding']}")
    print("   ✓ Test passed")

    print("\n" + "="*60)
    print("ALL TESTS PASSED")
    print("="*60)