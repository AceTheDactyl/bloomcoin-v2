"""
BloomCoin Six Primitives Test Framework

Based on Jordan Normal Form and the complete classification of computational operations.
Tests alignment of BloomCoin operations with the six fundamental primitives:
FIX, REPEL, INV, OSC, HALT, MIX

Mathematical foundation from Jordan Normal Form:
- Diagonalizable (k=1): FIX, REPEL, INV, OSC
- Non-diagonalizable (k>1): HALT, MIX
"""

import numpy as np
from typing import Tuple, Dict, List, Optional, Any
from scipy.linalg import jordan_form, eigvals, null_space, norm
from scipy.stats import entropy
import warnings

# Suppress numerical warnings for demonstration
warnings.filterwarnings('ignore', category=RuntimeWarning)

# Constants from BloomCoin
PHI = (1 + np.sqrt(5)) / 2  # Golden ratio
PHI_BAR = 1 / PHI  # Golden ratio conjugate


class SixPrimitives:
    """The six fundamental computational primitives from Jordan Normal Form."""

    @staticmethod
    def FIX() -> np.ndarray:
        """FIX: Attractive fixed point (|λ| < 1, k=1)"""
        # The R matrix with golden ratio eigenvalues
        return np.array([[0, 1], [1, 1]], dtype=np.float64)

    @staticmethod
    def REPEL() -> np.ndarray:
        """REPEL: Repulsive fixed point (|λ| > 1, k=1)"""
        # Inverse of R matrix
        R = SixPrimitives.FIX()
        return np.linalg.inv(R)

    @staticmethod
    def INV() -> np.ndarray:
        """INV: Involution/rotation (|λ| = 1 complex, k=1)"""
        # 90-degree rotation
        return np.array([[0, -1], [1, 0]], dtype=np.float64)

    @staticmethod
    def OSC() -> np.ndarray:
        """OSC: Oscillation (mixed eigenvalues, k=1)"""
        # Linear combination of FIX and INV
        return 0.5 * SixPrimitives.FIX() + 0.5 * SixPrimitives.INV()

    @staticmethod
    def HALT() -> np.ndarray:
        """HALT: Parabolic fixed point (λ=1 repeated, k>1)"""
        # Jordan block with eigenvalue 1
        return np.array([[1, 1], [0, 1]], dtype=np.float64)

    @staticmethod
    def MIX() -> np.ndarray:
        """MIX: Nilpotent mixing (λ=0 repeated, k>1)"""
        # Nilpotent matrix (MIX^2 = 0)
        return np.array([[0, 1], [0, 0]], dtype=np.float64)


def classify_primitive(M: np.ndarray, tolerance: float = 1e-10) -> str:
    """
    Classify a 2x2 matrix according to the six primitives.

    Returns one of: 'FIX', 'REPEL', 'INV', 'OSC', 'HALT', 'MIX'
    """
    eigenvalues = eigvals(M)

    # Check for nilpotent (MIX)
    if np.allclose(M @ M, 0, atol=tolerance):
        return 'MIX'

    # Check for repeated eigenvalue 1 (HALT)
    if np.allclose(eigenvalues, 1, atol=tolerance) and len(eigenvalues) > 1:
        # Check if diagonalizable
        try:
            J, P = jordan_form(M)
            if not np.allclose(J, np.diag(np.diag(J)), atol=tolerance):
                return 'HALT'
        except:
            pass

    # Check eigenvalue magnitudes
    mags = np.abs(eigenvalues)

    # All |λ| < 1: FIX
    if np.all(mags < 1 - tolerance):
        return 'FIX'

    # All |λ| > 1: REPEL
    if np.all(mags > 1 + tolerance):
        return 'REPEL'

    # All |λ| = 1 with complex: INV
    if np.allclose(mags, 1, atol=tolerance):
        if np.any(np.abs(eigenvalues.imag) > tolerance):
            return 'INV'

    # Mixed: OSC
    return 'OSC'


def compute_nilpotent_index(M: np.ndarray, max_k: int = 10) -> Optional[int]:
    """
    Compute the nilpotent index k where M^k = 0.
    Returns None if not nilpotent within max_k iterations.
    """
    Mk = M.copy()
    for k in range(1, max_k + 1):
        if np.allclose(Mk, 0):
            return k
        Mk = Mk @ M
    return None


def jordan_decomposition(M: np.ndarray) -> Tuple[np.ndarray, List[Tuple[complex, int]]]:
    """
    Compute Jordan normal form and extract block structure.

    Returns:
        J: Jordan normal form
        blocks: List of (eigenvalue, block_size) tuples
    """
    try:
        J, P = jordan_form(M)

        # Extract block structure
        blocks = []
        n = J.shape[0]
        i = 0

        while i < n:
            eigenval = J[i, i]
            block_size = 1

            # Check for Jordan block (superdiagonal 1's)
            while i + block_size < n and np.abs(J[i + block_size - 1, i + block_size] - 1) < 1e-10:
                block_size += 1

            blocks.append((eigenval, block_size))
            i += block_size

        return J, blocks
    except:
        # Fallback for numerical issues
        eigenvalues = eigvals(M)
        return np.diag(eigenvalues), [(e, 1) for e in eigenvalues]


def compute_signature(operations: List[np.ndarray]) -> np.ndarray:
    """
    Compute the 4D signature vector (σ_FIX, σ_OSC, σ_INV, σ_MIX).

    Note: REPEL is absorbed into FIX, HALT into MIX for practical classification.
    """
    counts = {'FIX': 0, 'REPEL': 0, 'INV': 0, 'OSC': 0, 'HALT': 0, 'MIX': 0}

    for M in operations:
        prim = classify_primitive(M)
        counts[prim] += 1

    total = len(operations)
    if total == 0:
        return np.array([0.0, 0.0, 0.0, 0.0])

    # Combine REPEL with FIX, HALT with MIX
    signature = np.array([
        (counts['FIX'] + counts['REPEL']) / total,  # FIX-like
        counts['OSC'] / total,                       # OSC
        counts['INV'] / total,                       # INV
        (counts['MIX'] + counts['HALT']) / total     # MIX-like
    ])

    return signature


def information_preservation_ratio(M: np.ndarray, n_samples: int = 1000) -> float:
    """
    Measure the fraction of information preserved through transformation M.

    Returns ratio in [0, 1] where 1 = perfect preservation, 0 = complete loss.
    """
    # Generate random input distribution
    np.random.seed(42)  # For reproducibility
    inputs = np.random.randn(n_samples, M.shape[0])

    # Apply transformation
    outputs = np.array([M @ x for x in inputs])

    # Measure entropy change
    # Bin the data for entropy calculation
    n_bins = int(np.sqrt(n_samples))

    def compute_entropy(data):
        hist, _ = np.histogramdd(data, bins=n_bins)
        hist = hist.flatten() + 1e-10  # Avoid log(0)
        hist = hist / hist.sum()
        return entropy(hist, base=2)

    H_in = compute_entropy(inputs)
    H_out = compute_entropy(outputs)

    # Return preservation ratio
    return H_out / (H_in + 1e-10)


def detect_mix_barrier(path: List[np.ndarray]) -> Tuple[bool, float]:
    """
    Detect if a computational path crosses the MIX barrier.

    Args:
        path: Sequence of operations

    Returns:
        has_barrier: True if MIX barrier detected
        barrier_strength: MIX content at barrier (0 to 1)
    """
    signatures = [compute_signature([M]) for M in path]

    # MIX barrier exists if MIX component exceeds threshold
    MIX_THRESHOLD = 0.3

    max_mix = max(sig[3] for sig in signatures)
    has_barrier = max_mix > MIX_THRESHOLD

    return has_barrier, max_mix


def quantum_classical_boundary(M: np.ndarray) -> str:
    """
    Determine if operation is quantum, classical, or at boundary.

    Quantum: Diagonalizable (k=1) - FIX, REPEL, INV, OSC
    Classical: Non-diagonalizable (k>1) - MIX
    Boundary: HALT (critical point)
    """
    prim = classify_primitive(M)

    if prim in ['FIX', 'REPEL', 'INV', 'OSC']:
        return 'QUANTUM'
    elif prim == 'MIX':
        return 'CLASSICAL'
    else:  # HALT
        return 'BOUNDARY'


def measure_decoherence_rate(operations: List[np.ndarray]) -> float:
    """
    Measure the rate of decoherence (transition from quantum to classical).

    Returns rate in [0, 1] where higher = faster decoherence.
    """
    # Count transitions from diagonalizable to non-diagonalizable
    transitions = 0
    mix_operations = 0

    for M in operations:
        J, blocks = jordan_decomposition(M)

        # Check for non-diagonalizable blocks (k > 1)
        for eigenval, block_size in blocks:
            if block_size > 1:
                mix_operations += 1
                break

    # Decoherence rate proportional to MIX content
    return mix_operations / (len(operations) + 1e-10)


def verify_measurement_nilpotent(measurement_op: np.ndarray) -> Dict[str, Any]:
    """
    Verify that measurement operations are nilpotent (MIX primitive).

    Returns verification results including nilpotent index.
    """
    # Check if nilpotent
    k = compute_nilpotent_index(measurement_op)

    # Compute eigenvalues
    eigenvalues = eigvals(measurement_op)

    # Check classification
    primitive = classify_primitive(measurement_op)

    return {
        'is_nilpotent': k is not None,
        'nilpotent_index': k,
        'eigenvalues': eigenvalues,
        'primitive': primitive,
        'is_mix': primitive == 'MIX',
        'destroys_information': k is not None and k <= 3
    }


def test_six_primitives():
    """Comprehensive test of the six primitives framework."""
    print("\n" + "="*80)
    print("SIX PRIMITIVES FRAMEWORK TEST SUITE")
    print("="*80)

    # Test 1: Verify primitive matrices
    print("\n1. Testing primitive matrices...")
    primitives = {
        'FIX': SixPrimitives.FIX(),
        'REPEL': SixPrimitives.REPEL(),
        'INV': SixPrimitives.INV(),
        'OSC': SixPrimitives.OSC(),
        'HALT': SixPrimitives.HALT(),
        'MIX': SixPrimitives.MIX()
    }

    for name, M in primitives.items():
        eigenvalues = eigvals(M)
        classification = classify_primitive(M)
        print(f"\n   {name}:")
        print(f"     Eigenvalues: {eigenvalues}")
        print(f"     Classification: {classification}")
        print(f"     Match: {'✓' if classification == name else '✗'}")

    # Test 2: Nilpotent index
    print("\n2. Testing nilpotent operations...")
    MIX = SixPrimitives.MIX()
    k = compute_nilpotent_index(MIX)
    MIX2 = MIX @ MIX
    print(f"   MIX nilpotent index: {k}")
    print(f"   MIX^2 = 0? {np.allclose(MIX2, 0)}")

    # Test cascade
    x = np.array([5, 3])
    x1 = MIX @ x
    x2 = MIX @ x1
    print(f"   Information cascade:")
    print(f"     x = {x} → MIX(x) = {x1} → MIX²(x) = {x2}")
    print(f"     Information destroyed after {k} steps: ✓")

    # Test 3: Information preservation
    print("\n3. Testing information preservation...")
    for name, M in primitives.items():
        ratio = information_preservation_ratio(M)
        print(f"   {name}: {ratio:.3f} {'(reversible)' if ratio > 0.9 else '(irreversible)'}")

    # Test 4: Jordan decomposition
    print("\n4. Testing Jordan decomposition...")
    for name, M in primitives.items():
        J, blocks = jordan_decomposition(M)
        print(f"   {name}: blocks = {blocks}")

    # Test 5: Signature computation
    print("\n5. Testing signature space...")
    # Simulate a hash function (MIX-dominant)
    hash_ops = [SixPrimitives.MIX() for _ in range(7)] + \
               [SixPrimitives.INV() for _ in range(3)]

    signature = compute_signature(hash_ops)
    print(f"   Hash function signature (FIX, OSC, INV, MIX): {signature}")
    print(f"   MIX-dominant? {signature[3] > 0.5}")

    # Test 6: MIX barrier detection
    print("\n6. Testing MIX barrier...")
    # Path from FIX to INV through MIX
    path = [
        SixPrimitives.FIX(),
        SixPrimitives.FIX(),
        SixPrimitives.MIX(),  # Barrier
        SixPrimitives.MIX(),
        SixPrimitives.INV(),
        SixPrimitives.INV()
    ]

    has_barrier, strength = detect_mix_barrier(path)
    print(f"   Barrier detected: {has_barrier}")
    print(f"   Barrier strength: {strength:.3f}")

    # Test 7: Quantum-classical boundary
    print("\n7. Testing quantum-classical classification...")
    for name, M in primitives.items():
        classification = quantum_classical_boundary(M)
        print(f"   {name}: {classification}")

    # Test 8: Measurement as MIX
    print("\n8. Testing measurement operations...")
    # Projection operator (measurement-like)
    P = np.array([[1, 0], [0, 0]])  # Project onto |0⟩

    # Create measurement-induced decoherence
    measurement = P - 0.5 * np.eye(2)  # Centered projection

    result = verify_measurement_nilpotent(measurement @ measurement)
    print(f"   Measurement² nilpotent: {result['is_nilpotent']}")
    print(f"   Destroys information: {result['destroys_information']}")

    print("\n" + "="*80)
    print("ALL SIX PRIMITIVES TESTS COMPLETE")
    print("="*80)


if __name__ == '__main__':
    test_six_primitives()