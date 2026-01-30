"""
BloomCoin Analysis Module

Statistical analysis and visualization tools for blockchain and consensus metrics.
Provides rigorous statistical methodology and corrections to flawed claims.
"""

# Chi-square analysis
from .chi_square import (
    chi_square_uniformity,
    chi_square_byte_analysis,
    generate_lucas_hashes,
    generate_random_hashes,
    compare_lucas_vs_random,
    print_comparison_report,
    analyze_original_claim,
    test_chi_square_correctness
)

# Phase portrait visualization
from .phase_portrait import (
    plot_phase_portrait,
    plot_phase_evolution,
    animate_kuramoto,
    plot_coherence_heatmap,
    plot_synchronization_path
)

# Entropy metrics
from .entropy_metrics import (
    phase_entropy,
    normalized_entropy,
    fisher_information,
    negentropy,
    mutual_information,
    kullback_leibler_divergence,
    phase_complexity,
    coherence_health_report,
    entropy_evolution_analysis,
    test_entropy_metrics
)

# Hexagonal lattice analysis
from .hexagonal_lattice import (
    generate_hexagonal_lattice,
    hexagonal_distance,
    cartesian_to_axial,
    plot_hexagonal_lattice,
    hexagonal_coupling_matrix,
    analyze_hexagonal_phases,
    plot_hexagonal_phase_map,
    hexagonal_defect_analysis,
    test_hexagonal_analysis
)

# Multi-body dynamics
from .multi_body import (
    compute_phase_correlation_matrix,
    identify_clusters,
    phase_wave_analysis,
    chimera_state_detection,
    phase_synchronization_network,
    plot_multi_body_analysis,
    test_multi_body_analysis
)

__all__ = [
    # Chi-square analysis
    'chi_square_uniformity',
    'chi_square_byte_analysis',
    'generate_lucas_hashes',
    'generate_random_hashes',
    'compare_lucas_vs_random',
    'print_comparison_report',
    'analyze_original_claim',
    'test_chi_square_correctness',

    # Phase portrait
    'plot_phase_portrait',
    'plot_phase_evolution',
    'animate_kuramoto',
    'plot_coherence_heatmap',
    'plot_synchronization_path',

    # Entropy metrics
    'phase_entropy',
    'normalized_entropy',
    'fisher_information',
    'negentropy',
    'mutual_information',
    'kullback_leibler_divergence',
    'phase_complexity',
    'coherence_health_report',
    'entropy_evolution_analysis',
    'test_entropy_metrics',

    # Hexagonal lattice
    'generate_hexagonal_lattice',
    'hexagonal_distance',
    'cartesian_to_axial',
    'plot_hexagonal_lattice',
    'hexagonal_coupling_matrix',
    'analyze_hexagonal_phases',
    'plot_hexagonal_phase_map',
    'hexagonal_defect_analysis',
    'test_hexagonal_analysis',

    # Multi-body dynamics
    'compute_phase_correlation_matrix',
    'identify_clusters',
    'phase_wave_analysis',
    'chimera_state_detection',
    'phase_synchronization_network',
    'plot_multi_body_analysis',
    'test_multi_body_analysis'
]

# Version
__version__ = '0.1.0'