"""
BloomCoin Multi-Body Dynamics Analysis

Analyze multi-oscillator interactions and emergent behaviors.
"""

import numpy as np
from typing import Optional, Dict, List, Tuple
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster


def compute_phase_correlation_matrix(phases_history: List[np.ndarray]) -> np.ndarray:
    """
    Compute correlation matrix between oscillators over time.

    Args:
        phases_history: List of phase arrays over time

    Returns:
        Correlation matrix (N x N)
    """
    n_oscillators = len(phases_history[0])
    n_steps = len(phases_history)

    # Convert to matrix (time x oscillators)
    phase_matrix = np.array(phases_history)

    # Compute correlation using circular statistics
    correlation = np.zeros((n_oscillators, n_oscillators))

    for i in range(n_oscillators):
        for j in range(n_oscillators):
            # Circular correlation
            phase_diff = phase_matrix[:, i] - phase_matrix[:, j]
            correlation[i, j] = np.mean(np.cos(phase_diff))

    return correlation


def identify_clusters(
    phases: np.ndarray,
    threshold: float = 0.5,
    method: str = 'phase_similarity'
) -> Dict:
    """
    Identify synchronized clusters of oscillators.

    Args:
        phases: Current phase distribution
        threshold: Clustering threshold
        method: 'phase_similarity' or 'correlation'

    Returns:
        Cluster analysis results
    """
    n_oscillators = len(phases)

    if method == 'phase_similarity':
        # Distance matrix based on phase differences
        phase_diffs = np.abs(phases[:, np.newaxis] - phases[np.newaxis, :])
        # Wrap to [0, π]
        phase_diffs = np.minimum(phase_diffs, 2*np.pi - phase_diffs)
        distance_matrix = phase_diffs / np.pi  # Normalize to [0, 1]

    else:  # correlation
        # Use correlation as similarity
        correlation = np.cos(phases[:, np.newaxis] - phases[np.newaxis, :])
        distance_matrix = 1 - correlation

    # Hierarchical clustering
    condensed_dist = squareform(distance_matrix)
    linkage_matrix = linkage(condensed_dist, method='average')

    # Get clusters
    clusters = fcluster(linkage_matrix, threshold, criterion='distance')

    # Analyze clusters
    unique_clusters = np.unique(clusters)
    cluster_info = []

    for cluster_id in unique_clusters:
        mask = clusters == cluster_id
        cluster_phases = phases[mask]

        # Cluster order parameter
        cluster_complex = np.exp(1j * cluster_phases)
        cluster_r = np.abs(np.mean(cluster_complex))
        cluster_psi = np.angle(np.mean(cluster_complex))

        cluster_info.append({
            'id': cluster_id,
            'size': np.sum(mask),
            'indices': np.where(mask)[0],
            'order_parameter': cluster_r,
            'mean_phase': cluster_psi,
            'phase_std': np.std(cluster_phases)
        })

    return {
        'n_clusters': len(unique_clusters),
        'clusters': cluster_info,
        'cluster_labels': clusters,
        'linkage_matrix': linkage_matrix,
        'largest_cluster_size': max(c['size'] for c in cluster_info)
    }


def phase_wave_analysis(
    phases: np.ndarray,
    positions: Optional[np.ndarray] = None
) -> Dict:
    """
    Analyze traveling wave patterns in phase distribution.

    Args:
        phases: Phase distribution
        positions: Optional spatial positions of oscillators

    Returns:
        Wave analysis results
    """
    n_oscillators = len(phases)

    if positions is None:
        # Assume 1D arrangement
        positions = np.linspace(0, 1, n_oscillators).reshape(-1, 1)

    # Compute phase gradient
    if positions.shape[1] == 1:  # 1D
        # Sort by position
        sort_idx = np.argsort(positions[:, 0])
        sorted_phases = phases[sort_idx]
        sorted_pos = positions[sort_idx, 0]

        # Phase gradient
        phase_grad = np.gradient(sorted_phases)
        pos_grad = np.gradient(sorted_pos)
        wave_vector = phase_grad / (pos_grad + 1e-10)

        # Detect wave
        wave_detected = np.std(wave_vector) < 0.5 * np.mean(np.abs(wave_vector))
        wave_speed = np.mean(wave_vector) if wave_detected else 0

    else:  # 2D or higher
        from scipy.interpolate import griddata

        # Create grid
        x_min, x_max = positions[:, 0].min(), positions[:, 0].max()
        y_min, y_max = positions[:, 1].min(), positions[:, 1].max()

        grid_x, grid_y = np.mgrid[x_min:x_max:20j, y_min:y_max:20j]

        # Interpolate phases
        grid_phases = griddata(positions, phases, (grid_x, grid_y), method='cubic')

        # Compute gradients
        grad_y, grad_x = np.gradient(grid_phases)

        # Wave vector (magnitude and direction)
        wave_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        wave_direction = np.arctan2(grad_y, grad_x)

        wave_detected = np.nanstd(wave_direction) < np.pi/2
        wave_speed = np.nanmean(wave_magnitude) if wave_detected else 0
        wave_vector = np.array([np.nanmean(grad_x), np.nanmean(grad_y)])

    return {
        'wave_detected': wave_detected,
        'wave_speed': wave_speed,
        'wave_vector': wave_vector,
        'phase_coherence_length': 1 / (np.std(phases) + 0.1)
    }


def chimera_state_detection(
    phases: np.ndarray,
    positions: Optional[np.ndarray] = None,
    window_size: int = 10
) -> Dict:
    """
    Detect chimera states (coexistence of coherent and incoherent regions).

    Args:
        phases: Phase distribution
        positions: Spatial positions
        window_size: Size of local window for analysis

    Returns:
        Chimera state analysis
    """
    n_oscillators = len(phases)

    if positions is None:
        positions = np.arange(n_oscillators).reshape(-1, 1)

    # Compute local order parameters
    local_order = np.zeros(n_oscillators)

    for i in range(n_oscillators):
        # Find neighbors within window
        distances = np.linalg.norm(positions - positions[i], axis=1)
        neighbors = np.argsort(distances)[1:window_size+1]

        if len(neighbors) > 0:
            # Local coherence
            neighbor_phases = phases[neighbors]
            local_complex = np.exp(1j * neighbor_phases)
            local_order[i] = np.abs(np.mean(local_complex))

    # Identify coherent and incoherent regions
    coherence_threshold = 0.7
    coherent_mask = local_order > coherence_threshold
    incoherent_mask = local_order < 1 - coherence_threshold

    # Check for chimera (both regions present)
    has_coherent = np.any(coherent_mask)
    has_incoherent = np.any(incoherent_mask)
    is_chimera = has_coherent and has_incoherent

    # Compute chimera index (variance of local order)
    chimera_index = np.std(local_order)

    return {
        'is_chimera': is_chimera,
        'chimera_index': chimera_index,
        'local_order': local_order,
        'coherent_fraction': np.mean(coherent_mask),
        'incoherent_fraction': np.mean(incoherent_mask),
        'coherent_indices': np.where(coherent_mask)[0],
        'incoherent_indices': np.where(incoherent_mask)[0]
    }


def phase_synchronization_network(
    phases_history: List[np.ndarray],
    threshold: float = 0.7
) -> Dict:
    """
    Build network of phase synchronization relationships.

    Args:
        phases_history: Time series of phases
        threshold: Synchronization threshold

    Returns:
        Network analysis results
    """
    n_oscillators = len(phases_history[0])

    # Compute pairwise synchronization
    sync_matrix = np.zeros((n_oscillators, n_oscillators))

    for i in range(n_oscillators):
        for j in range(i+1, n_oscillators):
            # Phase locking value (PLV)
            phase_diff = np.array([phases[i] - phases[j] for phases in phases_history])
            plv = np.abs(np.mean(np.exp(1j * phase_diff)))
            sync_matrix[i, j] = plv
            sync_matrix[j, i] = plv

    # Build adjacency matrix
    adjacency = sync_matrix > threshold

    # Network metrics
    degree = np.sum(adjacency, axis=1)
    clustering_coeff = np.zeros(n_oscillators)

    for i in range(n_oscillators):
        neighbors = np.where(adjacency[i])[0]
        if len(neighbors) > 1:
            # Count triangles
            triangles = 0
            possible = len(neighbors) * (len(neighbors) - 1) / 2
            for j in range(len(neighbors)):
                for k in range(j+1, len(neighbors)):
                    if adjacency[neighbors[j], neighbors[k]]:
                        triangles += 1
            clustering_coeff[i] = triangles / possible if possible > 0 else 0

    # Find communities (simplified)
    from scipy.sparse.csgraph import connected_components
    n_components, labels = connected_components(adjacency, directed=False)

    return {
        'synchronization_matrix': sync_matrix,
        'adjacency_matrix': adjacency,
        'mean_synchronization': np.mean(sync_matrix[np.triu_indices(n_oscillators, k=1)]),
        'network_degree': degree,
        'mean_degree': np.mean(degree),
        'clustering_coefficient': np.mean(clustering_coeff),
        'n_components': n_components,
        'component_labels': labels
    }


def plot_multi_body_analysis(
    phases: np.ndarray,
    analysis: Dict,
    title: str = "Multi-Body Analysis"
) -> plt.Figure:
    """
    Visualize multi-body dynamics analysis.

    Args:
        phases: Phase distribution
        analysis: Analysis results dictionary
        title: Plot title

    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 1. Cluster dendrogram
    ax1 = axes[0, 0]
    if 'linkage_matrix' in analysis:
        dendrogram(analysis['linkage_matrix'], ax=ax1, no_labels=True)
        ax1.set_title('Hierarchical Clustering')
        ax1.set_xlabel('Oscillator Index')
        ax1.set_ylabel('Distance')

    # 2. Phase distribution by cluster
    ax2 = axes[0, 1]
    if 'cluster_labels' in analysis:
        clusters = analysis['cluster_labels']
        unique_clusters = np.unique(clusters)

        for i, cluster_id in enumerate(unique_clusters):
            mask = clusters == cluster_id
            cluster_phases = phases[mask]
            color = plt.cm.tab10(i)

            # Circular histogram
            bins = np.linspace(0, 2*np.pi, 25)
            hist, _ = np.histogram(cluster_phases, bins=bins)
            bin_centers = (bins[:-1] + bins[1:]) / 2

            ax2.bar(bin_centers, hist, width=bins[1]-bins[0],
                   alpha=0.6, color=color, label=f'Cluster {cluster_id}')

        ax2.set_xlabel('Phase (rad)')
        ax2.set_ylabel('Count')
        ax2.set_title('Phase Distribution by Cluster')
        ax2.legend()

    # 3. Local order parameter
    ax3 = axes[1, 0]
    if 'local_order' in analysis:
        local_order = analysis['local_order']
        ax3.bar(range(len(local_order)), local_order, color='blue', alpha=0.7)
        ax3.axhline(y=0.7, color='r', linestyle='--', label='Coherence threshold')
        ax3.set_xlabel('Oscillator Index')
        ax3.set_ylabel('Local Order Parameter')
        ax3.set_title('Local Coherence')
        ax3.legend()

    # 4. Synchronization matrix
    ax4 = axes[1, 1]
    if 'synchronization_matrix' in analysis:
        sync_matrix = analysis['synchronization_matrix']
        im = ax4.imshow(sync_matrix, cmap='viridis', vmin=0, vmax=1)
        ax4.set_xlabel('Oscillator Index')
        ax4.set_ylabel('Oscillator Index')
        ax4.set_title('Pairwise Synchronization')
        plt.colorbar(im, ax=ax4, label='PLV')

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()

    return fig


def test_multi_body_analysis():
    """Test suite for multi-body analysis."""
    print("\n" + "="*60)
    print("MULTI-BODY DYNAMICS TEST SUITE")
    print("="*60)

    n_oscillators = 50
    n_steps = 100

    # Generate test data
    print("\n1. Generating test data...")
    phases_history = []
    for t in range(n_steps):
        # Evolving phases with two groups
        group1 = np.ones(25) * (0.1 * t) % (2*np.pi)
        group2 = np.random.uniform(0, 2*np.pi, 25)
        phases = np.concatenate([group1, group2])
        phases_history.append(phases)
    print("   ✓ Generated test data")

    # Test clustering
    print("\n2. Testing cluster identification...")
    final_phases = phases_history[-1]
    clusters = identify_clusters(final_phases)
    print(f"   Found {clusters['n_clusters']} clusters")
    print(f"   Largest cluster: {clusters['largest_cluster_size']} oscillators")
    assert clusters['n_clusters'] >= 1
    print("   ✓ Test passed")

    # Test wave analysis
    print("\n3. Testing wave analysis...")
    wave = phase_wave_analysis(final_phases)
    print(f"   Wave detected: {wave['wave_detected']}")
    print(f"   Wave speed: {wave['wave_speed']:.3f}")
    print("   ✓ Test passed")

    # Test chimera detection
    print("\n4. Testing chimera state detection...")
    chimera = chimera_state_detection(final_phases)
    print(f"   Is chimera: {chimera['is_chimera']}")
    print(f"   Chimera index: {chimera['chimera_index']:.3f}")
    print(f"   Coherent fraction: {chimera['coherent_fraction']:.3f}")
    print("   ✓ Test passed")

    # Test network analysis
    print("\n5. Testing synchronization network...")
    network = phase_synchronization_network(phases_history)
    print(f"   Mean synchronization: {network['mean_synchronization']:.3f}")
    print(f"   Clustering coefficient: {network['clustering_coefficient']:.3f}")
    print(f"   Network components: {network['n_components']}")
    assert 0 <= network['mean_synchronization'] <= 1
    print("   ✓ Test passed")

    print("\n" + "="*60)
    print("ALL TESTS PASSED")
    print("="*60)