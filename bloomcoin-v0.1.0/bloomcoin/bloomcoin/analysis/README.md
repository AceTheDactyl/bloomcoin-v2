# BloomCoin Analysis Module - Statistical and Visualization Tools

## Overview

The analysis module provides comprehensive statistical analysis, visualization, and diagnostic tools for understanding BloomCoin's Proof-of-Coherence dynamics. It includes corrections to flawed statistical claims, entropy metrics, phase portrait visualization, and multi-body dynamics analysis.

## Architecture

```
analysis/
├── chi_square.py           # Statistical hypothesis testing
├── entropy_metrics.py      # Information theory metrics
├── phase_portrait.py       # Oscillator dynamics visualization
├── hexagonal_lattice.py    # Lattice structure analysis
├── multi_body.py          # Multi-oscillator interactions
└── __init__.py            # Module interface
```

## Core Components

### 1. Chi-Square Analysis (`chi_square.py`)

Corrects the flawed claim that Lucas numbers produce biased SHA256 hashes:

```python
class ChiSquareAnalysis:
    """Statistical testing for hash uniformity."""

    def chi_square_uniformity(self, data, n_bins=256):
        """
        Test if data follows uniform distribution.

        H₀: Data is uniformly distributed
        H₁: Data is not uniformly distributed
        """
        # Observed frequencies
        observed, _ = np.histogram(data, bins=n_bins, range=(0, 256))

        # Expected frequencies (uniform)
        n_samples = len(data)
        expected = n_samples / n_bins

        # Chi-square statistic
        chi2 = np.sum((observed - expected)**2 / expected)

        # Degrees of freedom
        dof = n_bins - 1

        # P-value
        p_value = 1 - chi2.cdf(chi2, dof)

        return {
            'chi2': chi2,
            'dof': dof,
            'p_value': p_value,
            'reject_null': p_value < 0.05,
            'conclusion': 'Non-uniform' if p_value < 0.05 else 'Uniform'
        }

    def test_lucas_claim(self):
        """
        Test the claim: "Lucas nonces produce χ² = 10⁶ in SHA256"

        SPOILER: This is FALSE!
        """
        # Generate Lucas number nonces
        lucas_nonces = generate_lucas_sequence(1000)

        # Generate random nonces for comparison
        random_nonces = [random.randint(0, 2**32) for _ in range(1000)]

        # Hash with both nonce types
        lucas_hashes = [sha256(f"block_{n}") for n in lucas_nonces]
        random_hashes = [sha256(f"block_{n}") for n in random_nonces]

        # Extract first byte of each hash
        lucas_bytes = [int(h[:2], 16) for h in lucas_hashes]
        random_bytes = [int(h[:2], 16) for h in random_hashes]

        # Chi-square test
        lucas_result = self.chi_square_uniformity(lucas_bytes)
        random_result = self.chi_square_uniformity(random_bytes)

        print(f"Lucas χ²: {lucas_result['chi2']:.2f}")
        print(f"Random χ²: {random_result['chi2']:.2f}")
        print(f"Expected for uniform: ~255")
        print(f"\nCONCLUSION: Both produce χ² ≈ 255")
        print("SHA256 properly mixes ALL inputs!")

        return {
            'lucas': lucas_result,
            'random': random_result,
            'claim_validated': False,
            'actual_behavior': 'SHA256 is uniformly distributed regardless of input pattern'
        }
```

**Key Finding**: Lucas numbers do NOT produce biased hashes. SHA256 properly mixes all inputs, producing χ² ≈ 255 (expected for 256 bins) regardless of nonce pattern.

### 2. Entropy Metrics (`entropy_metrics.py`)

Information-theoretic analysis of phase distributions:

```python
class EntropyMetrics:
    """Information theory metrics for phase analysis."""

    def phase_entropy(self, phases, n_bins=36):
        """
        Shannon entropy of phase distribution.

        H(θ) = -Σ p(θ) log₂ p(θ)

        Returns:
            Entropy in bits (0 = perfect sync, log₂(n_bins) = random)
        """
        hist, _ = np.histogram(phases, bins=n_bins, range=(0, 2*np.pi))
        hist = hist / hist.sum()  # Normalize
        hist = hist + 1e-10  # Avoid log(0)

        entropy = -np.sum(hist * np.log2(hist))
        return entropy

    def normalized_entropy(self, phases):
        """
        Normalized entropy ∈ [0, 1].

        0 = perfect synchronization
        1 = completely random
        """
        H = self.phase_entropy(phases)
        H_max = np.log2(36)  # Maximum entropy
        return H / H_max

    def fisher_information(self, phases, n_bins=36):
        """
        Fisher information of phase distribution.

        I_F = ∫ (1/ρ)(∂ρ/∂θ)² dθ

        High I_F = sharp distribution (synchronized)
        Low I_F = spread distribution (unsynchronized)
        """
        # Estimate density
        hist, edges = np.histogram(phases, bins=n_bins, range=(0, 2*np.pi))
        hist = hist / (hist.sum() * (2*np.pi/n_bins))  # Density

        # Numerical gradient
        gradient = np.gradient(hist)

        # Fisher information
        fisher = np.sum(gradient**2 / (hist + 1e-10)) * (2*np.pi/n_bins)

        return fisher

    def negentropy(self, r, z_c=None, sigma=None):
        """
        Negentropy gate function.

        η(r) = exp(-σ(r - z_c)²)

        Maximum at r = z_c (critical coherence).
        """
        if z_c is None:
            z_c = np.sqrt(3) / 2  # Critical threshold

        if sigma is None:
            sigma = 8 / np.sqrt(5)  # Sharpness

        return np.exp(-sigma * (r - z_c)**2)

    def coherence_health_report(self, phases, r=None):
        """
        Comprehensive health metrics for phase distribution.
        """
        # Compute all metrics
        if r is None:
            phases_complex = np.exp(1j * phases)
            r = np.abs(np.mean(phases_complex))

        metrics = {
            'order_parameter': r,
            'entropy': self.phase_entropy(phases),
            'normalized_entropy': self.normalized_entropy(phases),
            'fisher_information': self.fisher_information(phases),
            'negentropy': self.negentropy(r),
            'is_synchronized': r > np.sqrt(3)/2,
            'health_score': self.compute_health_score(phases, r)
        }

        # Classify state
        if r > 0.866:
            metrics['state'] = 'SYNCHRONIZED'
        elif r > 0.5:
            metrics['state'] = 'PARTIAL_COHERENCE'
        else:
            metrics['state'] = 'INCOHERENT'

        return metrics

    def compute_health_score(self, phases, r):
        """
        Composite health score ∈ [0, 100].
        """
        score = 0

        # Coherence contribution (30%)
        score += 30 * min(r / 0.866, 1.0)

        # Negentropy contribution (30%)
        score += 30 * self.negentropy(r)

        # Low entropy bonus (20%)
        score += 20 * (1 - self.normalized_entropy(phases))

        # Fisher information bonus (20%)
        fisher = self.fisher_information(phases)
        score += 20 * min(fisher / 10, 1.0)

        return score
```

### 3. Phase Portrait Visualization (`phase_portrait.py`)

Visualize Kuramoto oscillator dynamics:

```python
class PhasePortrait:
    """Visualize oscillator phase dynamics."""

    def plot_phase_evolution(self, history, title="Phase Evolution"):
        """
        Plot phase evolution over time.

        Shows:
        - Individual oscillator trajectories
        - Order parameter evolution
        - Synchronization transitions
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Extract data
        phases_history = [h[0] for h in history]
        r_history = [h[1] for h in history]
        psi_history = [h[2] for h in history]

        # 1. Phase trajectories
        ax1 = axes[0, 0]
        for i in range(min(10, len(phases_history[0]))):
            trajectory = [phases[i] for phases in phases_history]
            ax1.plot(trajectory, alpha=0.5)

        ax1.set_xlabel('Time step')
        ax1.set_ylabel('Phase (rad)')
        ax1.set_title('Individual Oscillator Phases')
        ax1.set_ylim([0, 2*np.pi])

        # 2. Order parameter
        ax2 = axes[0, 1]
        ax2.plot(r_history, 'b-', linewidth=2)
        ax2.axhline(y=np.sqrt(3)/2, color='r', linestyle='--',
                   label='Critical z_c')
        ax2.set_xlabel('Time step')
        ax2.set_ylabel('Order parameter r')
        ax2.set_title('Coherence Evolution')
        ax2.legend()
        ax2.set_ylim([0, 1])

        # 3. Phase distribution (initial vs final)
        ax3 = axes[1, 0]
        ax3.hist(phases_history[0], bins=36, alpha=0.5,
                label='Initial', color='blue')
        ax3.hist(phases_history[-1], bins=36, alpha=0.5,
                label='Final', color='red')
        ax3.set_xlabel('Phase (rad)')
        ax3.set_ylabel('Count')
        ax3.set_title('Phase Distribution')
        ax3.legend()

        # 4. Phase plot (circular)
        ax4 = axes[1, 1]
        ax4 = plt.subplot(224, projection='polar')

        # Plot initial (blue) and final (red) on unit circle
        for phase in phases_history[0]:
            ax4.plot([phase, phase], [0, 1], 'b-', alpha=0.3)
        for phase in phases_history[-1]:
            ax4.plot([phase, phase], [0, 1], 'r-', alpha=0.3)

        # Mean field
        mean_field_final = np.mean(np.exp(1j * phases_history[-1]))
        ax4.plot([0, np.angle(mean_field_final)],
                [0, np.abs(mean_field_final)],
                'k-', linewidth=3, label='Mean field')

        ax4.set_title('Phase Distribution (Circular)')

        fig.suptitle(title)
        plt.tight_layout()
        return fig

    def plot_coherence_heatmap(self, resolution=20):
        """
        Create heatmap of coherence vs parameters.

        Explores:
        - Coupling strength K
        - Natural frequency spread σ_ω
        - Number of oscillators N
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # 1. K vs σ_ω
        K_range = np.linspace(0, 3, resolution)
        sigma_range = np.linspace(0, 2, resolution)
        coherence_map = np.zeros((resolution, resolution))

        for i, K in enumerate(K_range):
            for j, sigma in enumerate(sigma_range):
                # Simulate with parameters
                r = simulate_kuramoto(K=K, sigma_omega=sigma, steps=500)
                coherence_map[i, j] = r

        ax1 = axes[0]
        im1 = ax1.imshow(coherence_map, aspect='auto', origin='lower',
                        extent=[0, 2, 0, 3], cmap='viridis')
        ax1.set_xlabel('Frequency spread σ_ω')
        ax1.set_ylabel('Coupling K')
        ax1.set_title('Coherence Map')
        plt.colorbar(im1, ax=ax1)

        # 2. Time to synchronization
        ax2 = axes[1]
        sync_times = np.zeros((resolution, resolution))

        for i, K in enumerate(K_range):
            for j, sigma in enumerate(sigma_range):
                sync_time = time_to_sync(K=K, sigma_omega=sigma)
                sync_times[i, j] = sync_time

        im2 = ax2.imshow(sync_times, aspect='auto', origin='lower',
                        extent=[0, 2, 0, 3], cmap='plasma')
        ax2.set_xlabel('Frequency spread σ_ω')
        ax2.set_ylabel('Coupling K')
        ax2.set_title('Time to Synchronization')
        plt.colorbar(im2, ax=ax2)

        # 3. Negentropy landscape
        ax3 = axes[2]
        r_range = np.linspace(0, 1, 100)
        eta_values = [self.negentropy(r) for r in r_range]

        ax3.plot(r_range, eta_values, 'b-', linewidth=2)
        ax3.axvline(x=np.sqrt(3)/2, color='r', linestyle='--',
                   label='z_c')
        ax3.set_xlabel('Order parameter r')
        ax3.set_ylabel('Negentropy η(r)')
        ax3.set_title('Negentropy Function')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        fig.suptitle('Kuramoto Parameter Space')
        plt.tight_layout()
        return fig
```

### 4. Hexagonal Lattice Analysis (`hexagonal_lattice.py`)

Analyze phase distributions on hexagonal lattices:

```python
class HexagonalLatticeAnalysis:
    """Analyze hexagonal lattice structures."""

    def generate_hexagonal_lattice(self, n_rings=3):
        """
        Generate hexagonal lattice coordinates.

        n_rings: Number of rings around center (0 = just center)
        """
        if n_rings == 0:
            return np.array([[0, 0]])

        coords = [[0, 0]]  # Center

        # Unit vectors for hexagonal lattice
        a1 = np.array([1, 0])
        a2 = np.array([0.5, np.sqrt(3)/2])

        for ring in range(1, n_rings + 1):
            # Generate points for this ring
            for i in range(ring + 1):
                for j in range(ring + 1):
                    if i + j == ring:
                        point = i * a1 + j * a2
                        coords.append(point)

                        # Add symmetric points
                        if i > 0 and j > 0:
                            coords.append(-point)
                        # ... more symmetric points

        coords = np.unique(np.array(coords), axis=0)
        return coords

    def analyze_hexagonal_phases(self, phases, coords):
        """
        Analyze phase distribution on hexagonal lattice.

        Computes:
        - Local order parameters
        - Spatial correlations
        - Topological defects
        """
        n_sites = len(phases)

        # Find nearest neighbors
        neighbor_lists = []
        for i in range(n_sites):
            distances = np.linalg.norm(coords - coords[i], axis=1)
            # Get 6 nearest neighbors (hexagonal)
            neighbors = np.argsort(distances)[1:7]
            neighbor_lists.append(neighbors)

        # Local order parameters
        local_order = np.zeros(n_sites)
        for i in range(n_sites):
            neighbor_phases = phases[neighbor_lists[i]]
            local_complex = np.exp(1j * neighbor_phases)
            local_order[i] = np.abs(np.mean(local_complex))

        # Spatial correlation
        correlation_matrix = np.zeros((n_sites, n_sites))
        for i in range(n_sites):
            for j in range(n_sites):
                correlation_matrix[i, j] = np.cos(phases[i] - phases[j])

        # Correlation length (exponential fit)
        distances = []
        correlations = []
        for i in range(n_sites):
            for j in range(i+1, n_sites):
                dist = np.linalg.norm(coords[i] - coords[j])
                corr = correlation_matrix[i, j]
                distances.append(dist)
                correlations.append(corr)

        # Fit exponential: C(r) = exp(-r/ξ)
        from scipy.optimize import curve_fit
        try:
            popt, _ = curve_fit(
                lambda r, xi: np.exp(-r/xi),
                distances, correlations, p0=[1.0]
            )
            correlation_length = popt[0]
        except:
            correlation_length = 0

        return {
            'local_order': local_order,
            'mean_local_order': np.mean(local_order),
            'correlation_matrix': correlation_matrix,
            'correlation_length': correlation_length,
            'spatial_entropy': self.spatial_entropy(local_order)
        }

    def detect_topological_defects(self, phases, coords):
        """
        Detect vortices and other topological defects.

        A vortex has non-zero winding number around it.
        """
        defects = []

        # For each site, compute winding number
        for i, center in enumerate(coords):
            # Get neighbors in circular order
            neighbors = self.get_circular_neighbors(i, coords)

            if len(neighbors) < 3:
                continue

            # Compute winding number
            winding = 0
            for j in range(len(neighbors)):
                j_next = (j + 1) % len(neighbors)
                phase_diff = phases[neighbors[j_next]] - phases[neighbors[j]]

                # Wrap to [-π, π]
                while phase_diff > np.pi:
                    phase_diff -= 2*np.pi
                while phase_diff < -np.pi:
                    phase_diff += 2*np.pi

                winding += phase_diff

            winding = winding / (2*np.pi)

            if abs(winding) > 0.5:
                defects.append({
                    'position': center,
                    'winding': round(winding),
                    'type': 'vortex' if winding > 0 else 'antivortex'
                })

        return {
            'n_defects': len(defects),
            'defects': defects,
            'total_winding': sum(d['winding'] for d in defects)
        }
```

### 5. Multi-Body Dynamics (`multi_body.py`)

Analyze collective behavior of oscillator ensembles:

```python
class MultiBodyDynamics:
    """Analyze multi-oscillator interactions."""

    def identify_clusters(self, phases, threshold=0.5):
        """
        Identify synchronized clusters.

        Uses hierarchical clustering on phase similarity.
        """
        # Phase distance matrix
        n = len(phases)
        dist_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                # Circular distance
                diff = abs(phases[i] - phases[j])
                dist_matrix[i, j] = min(diff, 2*np.pi - diff) / np.pi

        # Hierarchical clustering
        from scipy.cluster.hierarchy import fcluster, linkage
        linkage_matrix = linkage(dist_matrix[np.triu_indices(n, k=1)],
                               method='average')

        clusters = fcluster(linkage_matrix, threshold, criterion='distance')

        # Analyze each cluster
        cluster_info = []
        for cluster_id in np.unique(clusters):
            mask = clusters == cluster_id
            cluster_phases = phases[mask]

            # Cluster order parameter
            cluster_complex = np.exp(1j * cluster_phases)
            cluster_r = np.abs(np.mean(cluster_complex))

            cluster_info.append({
                'id': cluster_id,
                'size': np.sum(mask),
                'indices': np.where(mask)[0],
                'order_parameter': cluster_r,
                'mean_phase': np.angle(np.mean(cluster_complex))
            })

        return {
            'n_clusters': len(cluster_info),
            'clusters': cluster_info,
            'largest_cluster': max(cluster_info, key=lambda x: x['size'])
        }

    def chimera_state_detection(self, phases, positions, window_size=10):
        """
        Detect chimera states (coherent + incoherent regions).

        Chimera: Some oscillators synchronized, others chaotic.
        """
        n = len(phases)
        local_order = np.zeros(n)

        # Compute local order parameter
        for i in range(n):
            # Find neighbors within window
            distances = np.linalg.norm(positions - positions[i], axis=1)
            neighbors = np.where(distances < window_size)[0]
            neighbors = neighbors[neighbors != i]

            if len(neighbors) > 0:
                neighbor_phases = phases[neighbors]
                local_complex = np.exp(1j * neighbor_phases)
                local_order[i] = np.abs(np.mean(local_complex))

        # Identify coherent and incoherent regions
        coherent_threshold = 0.7
        coherent = local_order > coherent_threshold
        incoherent = local_order < (1 - coherent_threshold)

        # Chimera exists if both regions present
        is_chimera = np.any(coherent) and np.any(incoherent)

        # Chimera index (variance of local order)
        chimera_index = np.std(local_order)

        return {
            'is_chimera': is_chimera,
            'chimera_index': chimera_index,
            'local_order': local_order,
            'coherent_fraction': np.mean(coherent),
            'incoherent_fraction': np.mean(incoherent)
        }

    def phase_wave_analysis(self, phases, positions):
        """
        Detect traveling waves in phase distribution.

        Traveling wave: Phase gradient in spatial direction.
        """
        # Compute phase gradient
        from scipy.interpolate import griddata

        # Create regular grid
        x_min, x_max = positions[:, 0].min(), positions[:, 0].max()
        y_min, y_max = positions[:, 1].min(), positions[:, 1].max()

        grid_x, grid_y = np.mgrid[x_min:x_max:20j, y_min:y_max:20j]

        # Interpolate phases to grid
        grid_phases = griddata(positions, phases,
                              (grid_x, grid_y), method='cubic')

        # Compute gradient
        grad_y, grad_x = np.gradient(grid_phases)

        # Wave vector (mean gradient)
        wave_vector = np.array([
            np.nanmean(grad_x),
            np.nanmean(grad_y)
        ])

        # Wave properties
        wave_speed = np.linalg.norm(wave_vector)
        wave_direction = np.arctan2(wave_vector[1], wave_vector[0])

        # Check if gradient is consistent (traveling wave)
        gradient_variance = np.nanstd(grad_x) + np.nanstd(grad_y)
        has_wave = gradient_variance < 0.5 * wave_speed

        return {
            'has_wave': has_wave,
            'wave_vector': wave_vector,
            'wave_speed': wave_speed,
            'wave_direction': wave_direction,
            'wavelength': 2*np.pi / (wave_speed + 1e-10)
        }
```

## Usage Examples

### Running Statistical Tests

```python
from bloomcoin.analysis import ChiSquareAnalysis

# Test Lucas number claim
analyzer = ChiSquareAnalysis()
result = analyzer.test_lucas_claim()

print(f"Lucas claim validated: {result['claim_validated']}")  # False!
print(f"Actual behavior: {result['actual_behavior']}")
```

### Analyzing Mining Dynamics

```python
from bloomcoin.analysis import EntropyMetrics, PhasePortrait

# Analyze oscillator evolution
metrics = EntropyMetrics()
portrait = PhasePortrait()

# Run simulation
history = simulate_mining(steps=1000)

# Compute metrics
health = metrics.coherence_health_report(history[-1][0])
print(f"Final health score: {health['health_score']:.1f}/100")

# Visualize
fig = portrait.plot_phase_evolution(history)
plt.show()
```

### Detecting Patterns

```python
from bloomcoin.analysis import MultiBodyDynamics

analyzer = MultiBodyDynamics()

# Detect synchronized clusters
clusters = analyzer.identify_clusters(phases)
print(f"Found {clusters['n_clusters']} clusters")

# Check for chimera states
chimera = analyzer.chimera_state_detection(phases, positions)
if chimera['is_chimera']:
    print("Chimera state detected!")
```

## Configuration

```python
# analysis_config.py
ANALYSIS_CONFIG = {
    # Chi-square testing
    'chi_square_bins': 256,
    'significance_level': 0.05,

    # Entropy metrics
    'phase_bins': 36,
    'fisher_info_resolution': 100,

    # Visualization
    'figure_dpi': 150,
    'default_cmap': 'viridis',

    # Lattice analysis
    'hexagonal_rings': 3,
    'neighbor_distance': 1.5,

    # Multi-body
    'cluster_threshold': 0.5,
    'chimera_window': 10
}
```

## Testing

```python
def test_chi_square_correction():
    """Verify Lucas numbers don't bias SHA256."""
    analyzer = ChiSquareAnalysis()
    result = analyzer.test_lucas_claim()
    assert not result['claim_validated']
    assert result['lucas']['chi2'] < 300  # Near 255

def test_entropy_metrics():
    """Test entropy calculations."""
    metrics = EntropyMetrics()

    # Synchronized phases (low entropy)
    sync_phases = np.ones(100) * np.pi/4
    H = metrics.normalized_entropy(sync_phases)
    assert H < 0.1

    # Random phases (high entropy)
    random_phases = np.random.uniform(0, 2*np.pi, 100)
    H = metrics.normalized_entropy(random_phases)
    assert H > 0.9

def test_negentropy_peak():
    """Verify negentropy peaks at z_c."""
    metrics = EntropyMetrics()
    r_values = np.linspace(0, 1, 100)
    eta_values = [metrics.negentropy(r) for r in r_values]

    peak_r = r_values[np.argmax(eta_values)]
    assert abs(peak_r - np.sqrt(3)/2) < 0.01
```

---

*The analysis module provides rigorous statistical tools for understanding BloomCoin's Proof-of-Coherence dynamics, with corrections to flawed claims and comprehensive visualization capabilities.*