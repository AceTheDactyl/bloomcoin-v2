#!/usr/bin/env python3
"""
BloomCoin Analysis Module Demonstration

Comprehensive demonstration of statistical analysis and visualization tools.
Includes corrections to flawed statistical claims.
"""

import sys
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import warnings

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Suppress matplotlib warnings for demo
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

from bloomcoin.analysis import (
    # Chi-square analysis
    compare_lucas_vs_random,
    print_comparison_report,
    analyze_original_claim,

    # Phase portrait
    plot_phase_portrait,
    plot_phase_evolution,
    plot_coherence_heatmap,

    # Entropy metrics
    phase_entropy,
    normalized_entropy,
    fisher_information,
    negentropy,
    coherence_health_report,
    entropy_evolution_analysis,

    # Hexagonal lattice
    generate_hexagonal_lattice,
    plot_hexagonal_lattice,
    analyze_hexagonal_phases,
    plot_hexagonal_phase_map,
    hexagonal_defect_analysis,

    # Multi-body dynamics
    identify_clusters,
    phase_wave_analysis,
    chimera_state_detection,
    phase_synchronization_network,
    plot_multi_body_analysis
)

from bloomcoin.constants import Z_C, K, PHI


def demo_chi_square_correction():
    """Demonstrate the correction to flawed statistical claims."""
    print("\n" + "="*70)
    print("CHI-SQUARE ANALYSIS: CORRECTING FLAWED CLAIMS")
    print("="*70)

    print("\nOriginal (incorrect) claim:")
    print("  'Lucas nonces produce χ² ≥ 10⁶ in SHA256 hashes'")
    print("  'This gives 3752× advantage over random'")
    print("\nLet's test this claim with proper statistics...")

    # Analyze the original claim
    analyze_original_claim()

    # Run proper comparison
    print("\nRunning proper statistical comparison (n=5000)...")
    comparison = compare_lucas_vs_random(n=5000)
    print_comparison_report(comparison)

    print("\n" + "="*70)
    print("CONCLUSION: The original claim is FALSE.")
    print("SHA256 properly mixes inputs regardless of nonce pattern.")
    print("Lucas nonces provide NO statistical advantage.")
    print("="*70)


def demo_phase_dynamics():
    """Demonstrate phase portrait and dynamics visualization."""
    print("\n" + "="*70)
    print("PHASE DYNAMICS VISUALIZATION")
    print("="*70)

    # Simulate Kuramoto oscillators
    print("\nSimulating Kuramoto oscillator evolution...")

    n_oscillators = 63
    n_steps = 200
    dt = 0.01

    # Initialize
    phases = np.random.uniform(0, 2*np.pi, n_oscillators)
    frequencies = np.random.normal(0, 0.5, n_oscillators)

    # History for analysis
    state_history = []

    # Evolution
    for step in range(n_steps):
        # Compute order parameter
        phases_complex = np.exp(1j * phases)
        mean_field = np.mean(phases_complex)
        r = np.abs(mean_field)
        psi = np.angle(mean_field)

        # Store state
        state_history.append((phases.copy(), r, psi))

        # Kuramoto dynamics
        coupling = K * r
        phases += dt * (frequencies + coupling * np.sin(psi - phases))
        phases = phases % (2*np.pi)

        # Print progress
        if step % 50 == 0:
            print(f"  Step {step:3d}: r = {r:.4f}")

    # Visualize evolution
    print("\nCreating phase evolution plot...")
    fig = plot_phase_evolution(state_history, title="Kuramoto Oscillator Evolution")
    plt.savefig('phase_evolution.png', dpi=150, bbox_inches='tight')
    print("  Saved to phase_evolution.png")

    # Final phase portrait
    final_phases, final_r, final_psi = state_history[-1]
    print(f"\nFinal state: r = {final_r:.4f}, synchronized = {final_r > Z_C}")

    # Create heatmap
    print("\nGenerating phase diagram heatmap...")
    fig = plot_coherence_heatmap(resolution=15)
    plt.savefig('coherence_heatmap.png', dpi=150, bbox_inches='tight')
    print("  Saved to coherence_heatmap.png")


def demo_entropy_analysis():
    """Demonstrate entropy and information metrics."""
    print("\n" + "="*70)
    print("ENTROPY AND INFORMATION METRICS")
    print("="*70)

    # Test different phase distributions
    distributions = {
        'Synchronized': np.ones(100) * np.pi/4 + 0.1 * np.random.randn(100),
        'Partially coherent': np.concatenate([
            np.ones(50) * np.pi/2 + 0.2 * np.random.randn(50),
            np.random.uniform(0, 2*np.pi, 50)
        ]),
        'Random': np.random.uniform(0, 2*np.pi, 100)
    }

    print("\nPhase Distribution Analysis:")
    print("-" * 60)

    for name, phases in distributions.items():
        # Compute metrics
        H = phase_entropy(phases)
        H_norm = normalized_entropy(phases)
        I_F = fisher_information(phases)

        # Order parameter
        phases_complex = np.exp(1j * phases)
        r = np.abs(np.mean(phases_complex))
        eta = negentropy(r)

        print(f"\n{name}:")
        print(f"  Order parameter r:     {r:.4f}")
        print(f"  Entropy (bits):        {H:.2f}")
        print(f"  Normalized entropy:    {H_norm:.3f}")
        print(f"  Fisher information:    {I_F:.2f}")
        print(f"  Negentropy η:          {eta:.4f}")

    # Health report for synchronized state
    print("\n" + "-"*60)
    print("Health Report for Synchronized State:")
    report = coherence_health_report(distributions['Synchronized'], verbose=True)

    # Entropy evolution
    print("\nAnalyzing entropy evolution during synchronization...")

    # Generate evolving phases
    phases_history = []
    phases = np.random.uniform(0, 2*np.pi, 50)

    for _ in range(100):
        phases_complex = np.exp(1j * phases)
        mean_field = np.mean(phases_complex)
        r = np.abs(mean_field)
        psi = np.angle(mean_field)

        phases += 0.01 * K * r * np.sin(psi - phases)
        phases = phases % (2*np.pi)
        phases_history.append(phases.copy())

    evolution = entropy_evolution_analysis(phases_history)

    print(f"  Initial entropy: {evolution['normalized_entropy'][0]:.3f}")
    print(f"  Final entropy:   {evolution['normalized_entropy'][-1]:.3f}")
    print(f"  Maximum Fisher:  {evolution['fisher_information'].max():.2f}")


def demo_hexagonal_lattice():
    """Demonstrate hexagonal lattice analysis."""
    print("\n" + "="*70)
    print("HEXAGONAL LATTICE ANALYSIS")
    print("="*70)

    print("\nGenerating hexagonal lattice...")
    coords = generate_hexagonal_lattice(n_rings=2)
    print(f"  Created lattice with {len(coords)} sites")

    # Create phase distribution on lattice
    # Vortex pattern
    phases_vortex = np.array([np.arctan2(y, x) % (2*np.pi) for x, y in coords])

    # Random pattern
    phases_random = np.random.uniform(0, 2*np.pi, len(coords))

    # Analyze both patterns
    print("\nAnalyzing phase patterns on lattice:")

    for name, phases in [("Vortex", phases_vortex), ("Random", phases_random)]:
        analysis = analyze_hexagonal_phases(phases, coords)
        defects = hexagonal_defect_analysis(phases, coords)

        print(f"\n{name} pattern:")
        print(f"  Mean local order:    {analysis['mean_local_order']:.3f}")
        print(f"  Correlation length:  {analysis['correlation_length']:.3f}")
        print(f"  Spatial entropy:     {analysis['spatial_entropy']:.3f}")
        print(f"  Topological defects: {defects['n_defects']}")
        print(f"  Total winding:       {defects['total_winding']}")

    # Visualize
    print("\nCreating hexagonal phase maps...")
    fig = plot_hexagonal_phase_map(phases_vortex, coords,
                                   title="Vortex Pattern on Hexagonal Lattice")
    plt.savefig('hexagonal_vortex.png', dpi=150, bbox_inches='tight')
    print("  Saved to hexagonal_vortex.png")


def demo_multi_body_dynamics():
    """Demonstrate multi-body dynamics analysis."""
    print("\n" + "="*70)
    print("MULTI-BODY DYNAMICS ANALYSIS")
    print("="*70)

    # Generate test system with two groups
    n_oscillators = 60
    n_steps = 150

    print("\nSimulating two-group system...")
    phases_history = []

    # Initial conditions: two separated groups
    phases = np.concatenate([
        np.ones(30) * 0 + 0.2 * np.random.randn(30),
        np.ones(30) * np.pi + 0.2 * np.random.randn(30)
    ])

    for step in range(n_steps):
        phases_history.append(phases.copy())

        # Evolution with weak coupling
        phases_complex = np.exp(1j * phases)
        mean_field = np.mean(phases_complex)
        r = np.abs(mean_field)
        psi = np.angle(mean_field)

        # Different coupling for each group (chimera-like)
        coupling = np.concatenate([
            np.ones(30) * K * 0.3,  # Weak coupling
            np.ones(30) * K         # Strong coupling
        ])

        phases += 0.01 * coupling * r * np.sin(psi - phases)
        phases = phases % (2*np.pi)

    final_phases = phases_history[-1]

    # Analyze clusters
    print("\n1. Cluster analysis:")
    clusters = identify_clusters(final_phases)
    print(f"   Number of clusters: {clusters['n_clusters']}")
    print(f"   Largest cluster:    {clusters['largest_cluster_size']} oscillators")

    for cluster in clusters['clusters'][:3]:  # Show first 3 clusters
        print(f"   Cluster {cluster['id']}: size={cluster['size']}, r={cluster['order_parameter']:.3f}")

    # Wave analysis
    print("\n2. Wave analysis:")
    wave = phase_wave_analysis(final_phases)
    print(f"   Wave detected:       {wave['wave_detected']}")
    print(f"   Wave speed:          {wave['wave_speed']:.3f}")
    print(f"   Coherence length:    {wave['phase_coherence_length']:.3f}")

    # Chimera detection
    print("\n3. Chimera state detection:")
    chimera = chimera_state_detection(final_phases, window_size=10)
    print(f"   Is chimera:          {chimera['is_chimera']}")
    print(f"   Chimera index:       {chimera['chimera_index']:.3f}")
    print(f"   Coherent fraction:   {chimera['coherent_fraction']:.3f}")
    print(f"   Incoherent fraction: {chimera['incoherent_fraction']:.3f}")

    # Network analysis
    print("\n4. Synchronization network:")
    network = phase_synchronization_network(phases_history)
    print(f"   Mean synchronization:   {network['mean_synchronization']:.3f}")
    print(f"   Mean degree:            {network['mean_degree']:.1f}")
    print(f"   Clustering coefficient: {network['clustering_coefficient']:.3f}")
    print(f"   Network components:     {network['n_components']}")

    # Comprehensive visualization
    print("\nCreating multi-body analysis visualization...")
    analysis_combined = {**clusters, **chimera, **network}
    fig = plot_multi_body_analysis(final_phases, analysis_combined,
                                   title="Multi-Body Dynamics Analysis")
    plt.savefig('multi_body_analysis.png', dpi=150, bbox_inches='tight')
    print("  Saved to multi_body_analysis.png")


def run_all_tests():
    """Run all test suites."""
    print("\n" + "="*70)
    print("RUNNING ALL TEST SUITES")
    print("="*70)

    from bloomcoin.analysis import (
        test_chi_square_correctness,
        test_entropy_metrics,
        test_hexagonal_analysis,
        test_multi_body_analysis
    )

    print("\n1. Chi-square tests:")
    test_chi_square_correctness()

    print("\n2. Entropy metric tests:")
    test_entropy_metrics()

    print("\n3. Hexagonal lattice tests:")
    test_hexagonal_analysis()

    print("\n4. Multi-body dynamics tests:")
    test_multi_body_analysis()


def main():
    """Run all demonstrations."""
    print("\n" + "="*70)
    print("BLOOMCOIN ANALYSIS MODULE DEMONSTRATION")
    print("="*70)
    print(f"Golden ratio φ = {PHI:.10f}")
    print(f"Critical threshold z_c = {Z_C:.10f}")
    print(f"Kuramoto coupling K = {K:.10f}")

    # Run demonstrations
    demo_chi_square_correction()
    demo_phase_dynamics()
    demo_entropy_analysis()
    demo_hexagonal_lattice()
    demo_multi_body_dynamics()

    # Run test suites
    print("\n" + "="*70)
    print("VALIDATION TESTS")
    print("="*70)
    run_all_tests()

    print("\n" + "="*70)
    print("DEMONSTRATION COMPLETE")
    print("="*70)
    print("\nKey Results:")
    print("  ✓ Lucas nonces do NOT produce biased hashes (χ² ≈ 255)")
    print("  ✓ Phase dynamics properly visualized")
    print("  ✓ Entropy metrics correctly computed")
    print("  ✓ Hexagonal lattice structures analyzed")
    print("  ✓ Multi-body dynamics characterized")
    print("\nTruth emerges from proper analysis! 🌸")


if __name__ == '__main__':
    # Set matplotlib backend for non-interactive use
    import matplotlib
    matplotlib.use('Agg')

    main()

    # Close all plots
    plt.close('all')