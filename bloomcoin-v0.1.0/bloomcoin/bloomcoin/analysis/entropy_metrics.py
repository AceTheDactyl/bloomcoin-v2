"""
BloomCoin Entropy Metrics

Information-theoretic analysis of phase distributions.
"""

import numpy as np
from typing import Optional, Dict, Tuple, List
from scipy import stats


def phase_entropy(phases: np.ndarray, n_bins: int = 36) -> float:
    """
    Compute Shannon entropy of phase distribution.

    H = -Σ p(θ) log₂ p(θ)

    Maximum entropy (uniform): log₂(n_bins)
    Minimum entropy (delta): 0

    Args:
        phases: Array of oscillator phases
        n_bins: Number of histogram bins

    Returns:
        Entropy in bits
    """
    hist, _ = np.histogram(phases, bins=n_bins, range=(0, 2*np.pi), density=True)

    # Normalize to probabilities
    hist = hist * (2 * np.pi / n_bins)
    hist = hist + 1e-10  # Avoid log(0)

    # Shannon entropy
    entropy = -np.sum(hist * np.log2(hist))

    return entropy


def normalized_entropy(phases: np.ndarray, n_bins: int = 36) -> float:
    """
    Entropy normalized to [0, 1] range.

    0 = all phases identical (perfect synchronization)
    1 = uniform distribution (no synchronization)

    Args:
        phases: Array of oscillator phases
        n_bins: Number of bins for discretization

    Returns:
        Normalized entropy
    """
    H = phase_entropy(phases, n_bins)
    H_max = np.log2(n_bins)
    return H / H_max


def fisher_information(phases: np.ndarray, n_bins: int = 36) -> float:
    """
    Compute Fisher Information of phase distribution.

    I_F = ∫ (1/ρ) (∂ρ/∂θ)² dθ

    High I_F: Sharp, localized distribution (synchronized)
    Low I_F: Spread distribution (unsynchronized)

    Args:
        phases: Array of oscillator phases
        n_bins: Number of histogram bins

    Returns:
        Fisher Information (dimensionless)
    """
    # Histogram-based density estimation
    hist, bin_edges = np.histogram(phases, bins=n_bins, range=(0, 2*np.pi), density=True)
    hist = np.maximum(hist, 1e-10)  # Avoid division by zero

    # Numerical gradient
    d_theta = bin_edges[1] - bin_edges[0]

    # Use central differences for gradient
    grad_rho = np.zeros_like(hist)
    grad_rho[1:-1] = (hist[2:] - hist[:-2]) / (2 * d_theta)
    grad_rho[0] = (hist[1] - hist[-1]) / (2 * d_theta)  # Periodic boundary
    grad_rho[-1] = (hist[0] - hist[-2]) / (2 * d_theta)

    # Fisher Information: I = ∫ (∂ρ/∂θ)² / ρ dθ
    integrand = grad_rho ** 2 / hist
    fisher = np.trapz(integrand, dx=d_theta)

    return float(fisher)


def negentropy(r: float, z_c: Optional[float] = None, sigma: Optional[float] = None) -> float:
    """
    Compute negentropy gate function η(r).

    η(r) = exp(-σ(r - z_c)²)

    This measures "health" of the phase distribution:
    - Maximum (η = 1) at r = z_c (optimal coherence)
    - Falls off for both lower and higher r

    Args:
        r: Order parameter
        z_c: Critical threshold (default from constants)
        sigma: Sharpness parameter (default from constants)

    Returns:
        η ∈ (0, 1]
    """
    from ..constants import Z_C, SIGMA

    if z_c is None:
        z_c = Z_C
    if sigma is None:
        sigma = SIGMA

    return np.exp(-sigma * (r - z_c) ** 2)


def mutual_information(phases1: np.ndarray, phases2: np.ndarray, n_bins: int = 20) -> float:
    """
    Compute mutual information between two phase distributions.

    MI(X,Y) = H(X) + H(Y) - H(X,Y)

    Measures how much information is shared between distributions.

    Args:
        phases1: First phase distribution
        phases2: Second phase distribution
        n_bins: Number of bins for discretization

    Returns:
        Mutual information in bits
    """
    # Discretize phases
    bins = np.linspace(0, 2*np.pi, n_bins+1)

    # Get bin indices
    x_bins = np.digitize(phases1, bins) - 1
    y_bins = np.digitize(phases2, bins) - 1

    # Clip to valid range
    x_bins = np.clip(x_bins, 0, n_bins-1)
    y_bins = np.clip(y_bins, 0, n_bins-1)

    # Compute 2D histogram
    hist_2d, _, _ = np.histogram2d(x_bins, y_bins, bins=[range(n_bins+1), range(n_bins+1)])

    # Normalize to probabilities
    pxy = hist_2d / hist_2d.sum()
    px = pxy.sum(axis=1)
    py = pxy.sum(axis=0)

    # Compute MI
    mi = 0.0
    for i in range(n_bins):
        for j in range(n_bins):
            if pxy[i,j] > 0 and px[i] > 0 and py[j] > 0:
                mi += pxy[i,j] * np.log2(pxy[i,j] / (px[i] * py[j]))

    return mi


def kullback_leibler_divergence(
    phases1: np.ndarray,
    phases2: np.ndarray,
    n_bins: int = 36
) -> float:
    """
    Compute KL divergence between two phase distributions.

    KL(P||Q) = Σ P(x) log(P(x)/Q(x))

    Measures how different two distributions are.
    Note: KL divergence is not symmetric.

    Args:
        phases1: Reference distribution
        phases2: Comparison distribution
        n_bins: Number of bins

    Returns:
        KL divergence in bits
    """
    # Get histograms
    hist1, bins = np.histogram(phases1, bins=n_bins, range=(0, 2*np.pi), density=True)
    hist2, _ = np.histogram(phases2, bins=bins, density=True)

    # Normalize
    hist1 = hist1 / hist1.sum()
    hist2 = hist2 / hist2.sum()

    # Add small epsilon to avoid log(0)
    eps = 1e-10
    hist1 = hist1 + eps
    hist2 = hist2 + eps

    # Compute KL divergence
    kl = np.sum(hist1 * np.log2(hist1 / hist2))

    return kl


def phase_complexity(phases: np.ndarray) -> Dict[str, float]:
    """
    Compute multiple complexity measures for phase distribution.

    Args:
        phases: Phase distribution

    Returns:
        Dictionary with various complexity metrics
    """
    # Order parameter
    phases_complex = np.exp(1j * phases)
    mean_complex = np.mean(phases_complex)
    r = np.abs(mean_complex)
    psi = np.angle(mean_complex)

    # Entropy measures
    H = phase_entropy(phases)
    H_norm = normalized_entropy(phases)
    I_F = fisher_information(phases)
    eta = negentropy(r)

    # Circular statistics
    circular_variance = 1 - r
    circular_std = np.sqrt(-2 * np.log(r)) if r > 0 else np.pi

    # Concentration parameter (von Mises)
    if r > 0.95:
        kappa = 1 / (1 - r)
    elif r > 0:
        # Approximation for moderate r
        kappa = r * (2 - r**2) / (1 - r**2)
    else:
        kappa = 0

    return {
        'order_parameter': r,
        'mean_phase': psi,
        'entropy_bits': H,
        'normalized_entropy': H_norm,
        'fisher_information': I_F,
        'negentropy': eta,
        'circular_variance': circular_variance,
        'circular_std': circular_std,
        'concentration_kappa': kappa
    }


def coherence_health_report(
    phases: np.ndarray,
    r: Optional[float] = None,
    verbose: bool = True
) -> Dict:
    """
    Comprehensive health report for phase distribution.

    Args:
        phases: Phase distribution
        r: Order parameter (computed if None)
        verbose: Print report to console

    Returns:
        Dict with health metrics
    """
    from ..constants import Z_C

    # Get all complexity measures
    metrics = phase_complexity(phases)

    if r is not None:
        metrics['order_parameter'] = r
        metrics['negentropy'] = negentropy(r)

    # Classification
    r = metrics['order_parameter']
    eta = metrics['negentropy']

    metrics['is_synchronized'] = r > Z_C
    metrics['is_healthy'] = eta > 0.5

    if r > Z_C:
        metrics['phase_state'] = 'synchronized'
    elif r > Z_C - 0.1:
        metrics['phase_state'] = 'near_critical'
    elif r > 0.5:
        metrics['phase_state'] = 'partially_coherent'
    else:
        metrics['phase_state'] = 'incoherent'

    # Health score (0-100)
    health_score = 0.0
    health_score += 30 * min(r / Z_C, 1.0)  # Coherence contribution
    health_score += 30 * eta  # Negentropy contribution
    health_score += 20 * (1 - metrics['normalized_entropy'])  # Low entropy
    health_score += 20 * min(metrics['fisher_information'] / 10, 1.0)  # High Fisher info
    metrics['health_score'] = health_score

    if verbose:
        print("\n" + "="*60)
        print("COHERENCE HEALTH REPORT")
        print("="*60)
        print(f"Phase state:        {metrics['phase_state'].upper()}")
        print(f"Order parameter:    r = {r:.4f}")
        print(f"Critical threshold: z_c = {Z_C:.4f}")
        print(f"Negentropy:         η = {eta:.4f}")
        print(f"Health score:       {health_score:.1f}/100")
        print("-"*60)
        print(f"Entropy (bits):     {metrics['entropy_bits']:.2f}")
        print(f"Normalized entropy: {metrics['normalized_entropy']:.3f}")
        print(f"Fisher information: {metrics['fisher_information']:.2f}")
        print(f"Circular std dev:   {metrics['circular_std']:.3f}")
        print("="*60)

    return metrics


def entropy_evolution_analysis(
    phases_history: List[np.ndarray],
    window_size: int = 10
) -> Dict[str, np.ndarray]:
    """
    Analyze entropy evolution over time.

    Args:
        phases_history: List of phase arrays over time
        window_size: Window for smoothing

    Returns:
        Dict with time series of entropy metrics
    """
    n_steps = len(phases_history)

    # Initialize arrays
    entropy_series = np.zeros(n_steps)
    fisher_series = np.zeros(n_steps)
    negentropy_series = np.zeros(n_steps)
    r_series = np.zeros(n_steps)

    for i, phases in enumerate(phases_history):
        # Order parameter
        phases_complex = np.exp(1j * phases)
        r = np.abs(np.mean(phases_complex))
        r_series[i] = r

        # Entropy metrics
        entropy_series[i] = normalized_entropy(phases)
        fisher_series[i] = fisher_information(phases)
        negentropy_series[i] = negentropy(r)

    # Compute derivatives
    entropy_rate = np.gradient(entropy_series)
    fisher_rate = np.gradient(fisher_series)

    # Smooth if requested
    if window_size > 1:
        from scipy.ndimage import uniform_filter1d
        entropy_series = uniform_filter1d(entropy_series, window_size, mode='nearest')
        fisher_series = uniform_filter1d(fisher_series, window_size, mode='nearest')
        entropy_rate = uniform_filter1d(entropy_rate, window_size, mode='nearest')
        fisher_rate = uniform_filter1d(fisher_rate, window_size, mode='nearest')

    return {
        'time': np.arange(n_steps),
        'order_parameter': r_series,
        'normalized_entropy': entropy_series,
        'fisher_information': fisher_series,
        'negentropy': negentropy_series,
        'entropy_rate': entropy_rate,
        'fisher_rate': fisher_rate
    }


def test_entropy_metrics():
    """Test suite for entropy metrics."""
    print("\n" + "="*60)
    print("ENTROPY METRICS TEST SUITE")
    print("="*60)

    # Test 1: Synchronized phases (low entropy)
    print("\n1. Testing synchronized distribution...")
    sync_phases = np.ones(100) * np.pi/4 + 0.1 * np.random.randn(100)
    H_sync = normalized_entropy(sync_phases)
    I_sync = fisher_information(sync_phases)
    print(f"   Normalized entropy: {H_sync:.3f} (should be low)")
    print(f"   Fisher information: {I_sync:.2f} (should be high)")
    assert H_sync < 0.5
    assert I_sync > 1.0
    print("   ✓ Test passed")

    # Test 2: Random phases (high entropy)
    print("\n2. Testing random distribution...")
    random_phases = np.random.uniform(0, 2*np.pi, 100)
    H_random = normalized_entropy(random_phases)
    I_random = fisher_information(random_phases)
    print(f"   Normalized entropy: {H_random:.3f} (should be high)")
    print(f"   Fisher information: {I_random:.2f} (should be low)")
    assert H_random > 0.7
    assert I_random < 5.0
    print("   ✓ Test passed")

    # Test 3: Negentropy peak at z_c
    print("\n3. Testing negentropy function...")
    from ..constants import Z_C

    r_values = np.linspace(0, 1, 100)
    eta_values = [negentropy(r) for r in r_values]
    max_idx = np.argmax(eta_values)
    r_max = r_values[max_idx]

    print(f"   Negentropy peaks at r = {r_max:.3f}")
    print(f"   Expected peak at z_c = {Z_C:.3f}")
    assert abs(r_max - Z_C) < 0.01
    print("   ✓ Test passed")

    # Test 4: Health report
    print("\n4. Testing health report...")
    test_phases = sync_phases
    report = coherence_health_report(test_phases, verbose=False)
    print(f"   Health score: {report['health_score']:.1f}/100")
    print(f"   Phase state: {report['phase_state']}")
    assert report['health_score'] > 50
    print("   ✓ Test passed")

    print("\n" + "="*60)
    print("ALL TESTS PASSED")
    print("="*60)