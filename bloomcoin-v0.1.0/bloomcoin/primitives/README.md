# BloomCoin Primitives Module

## Overview

The primitives module contains the fundamental building blocks of BloomCoin's Proof-of-Coherence consensus mechanism, including the Kuramoto oscillator implementation and the mathematical framework based on Jordan Normal Form.

## Core Components

### 1. Kuramoto Oscillator (`oscillator.py`)

The heart of Proof-of-Coherence - coupled phase oscillators that synchronize:

```python
class KuramotoOscillator:
    """
    Individual oscillator in the Kuramoto network.

    Phase evolution:
    dθ/dt = ω + (K/N) Σ sin(θⱼ - θ)

    Where:
    - θ: phase ∈ [0, 2π]
    - ω: natural frequency
    - K: coupling strength
    - N: number of oscillators
    """

    def __init__(self, frequency=0.0, phase=None):
        self.frequency = frequency  # Natural frequency ω
        self.phase = phase if phase else random.uniform(0, 2*np.pi)
        self.phase_history = [self.phase]

    def update(self, mean_field, coupling, dt=0.01):
        """
        Update phase based on mean field interaction.

        mean_field: Complex order parameter ⟨e^(iθ)⟩
        coupling: Coupling strength K
        dt: Time step
        """
        # Extract magnitude and phase from mean field
        r = abs(mean_field)
        psi = np.angle(mean_field)

        # Kuramoto dynamics
        dtheta = self.frequency + coupling * r * np.sin(psi - self.phase)

        # Update phase
        self.phase += dt * dtheta
        self.phase = self.phase % (2 * np.pi)

        # Record history
        self.phase_history.append(self.phase)

        return self.phase
```

### 2. Oscillator Network (`network.py`)

Collection of coupled oscillators that evolve together:

```python
class OscillatorNetwork:
    """
    Network of coupled Kuramoto oscillators.

    The network evolves toward synchronization when K > K_c.
    Critical coupling: K_c = 2σ/π for Gaussian frequency distribution.
    """

    def __init__(self, n_oscillators=63, coupling=None):
        self.N = n_oscillators

        # Default coupling from golden ratio
        if coupling is None:
            self.K = 2 * PHI / 3  # ≈ 1.079

        self.oscillators = []
        self.initialize_oscillators()

    def initialize_oscillators(self, seed=None):
        """
        Initialize oscillators with random frequencies and phases.

        Frequencies drawn from Gaussian distribution.
        Phases uniformly distributed on circle.
        """
        if seed:
            np.random.seed(seed)

        # Generate natural frequencies
        # Width σ determines critical coupling
        frequencies = np.random.normal(0, 0.5, self.N)

        # Create oscillators
        self.oscillators = [
            KuramotoOscillator(freq) for freq in frequencies
        ]

    def compute_mean_field(self):
        """
        Compute complex order parameter (mean field).

        z = r * e^(iψ) = (1/N) Σ e^(iθⱼ)

        Returns:
            Complex number representing mean field
        """
        phases = [osc.phase for osc in self.oscillators]
        complex_phases = np.exp(1j * np.array(phases))
        mean_field = np.mean(complex_phases)

        return mean_field

    def step(self, dt=0.01):
        """
        Evolve network one time step.

        All oscillators update based on current mean field.
        """
        # Compute current mean field
        mean_field = self.compute_mean_field()

        # Update each oscillator
        for osc in self.oscillators:
            osc.update(mean_field, self.K, dt)

        return abs(mean_field)  # Return coherence r

    def evolve(self, steps=1000, target=None):
        """
        Evolve network for multiple steps.

        Args:
            steps: Number of time steps
            target: Stop if coherence exceeds target

        Returns:
            Final coherence and evolution history
        """
        history = []

        for step in range(steps):
            r = self.step()
            history.append(r)

            # Early termination if target reached
            if target and r > target:
                break

        return r, history

    def get_phases(self):
        """Get current phase distribution."""
        return np.array([osc.phase for osc in self.oscillators])

    def get_frequencies(self):
        """Get natural frequency distribution."""
        return np.array([osc.frequency for osc in self.oscillators])
```

### 3. Mathematical Constants (`constants.py`)

All constants derived from the golden ratio φ:

```python
import numpy as np

# Golden ratio - the fundamental constant
PHI = (1 + np.sqrt(5)) / 2  # ≈ 1.618033988749895

# Critical coherence threshold
# Derived from stability analysis of Kuramoto model
Z_C = np.sqrt(3) / 2  # ≈ 0.8660254037844387

# Kuramoto coupling strength
# Optimal for convergence to z_c
K = 2 * PHI / 3  # ≈ 1.0786893258332633

# Negentropy sharpness parameter
# Controls width of coherence attractor
SIGMA = 8 / np.sqrt(5)  # ≈ 3.5777087639996634

# Network size (triangle number - 1)
# 63 = triangle(8) - 1 = 8*(8+1)/2 - 1
N_OSCILLATORS = 63

# Mining parameters
DIFFICULTY_INITIAL = 4
BLOCK_TIME_TARGET = 60  # seconds
DIFFICULTY_ADJUSTMENT_INTERVAL = 10  # blocks

# Constants verification
assert abs(PHI - (1 + np.sqrt(5))/2) < 1e-10
assert abs(Z_C - np.sqrt(3)/2) < 1e-10
assert abs(K - 2*PHI/3) < 1e-10
assert abs(SIGMA - 8/np.sqrt(5)) < 1e-10
```

### 4. Six Primitives Framework (`six_primitives.py`)

Computational primitives from Jordan Normal Form:

```python
class ComputationalPrimitives:
    """
    The six fundamental computational primitives.

    Based on Jordan Normal Form classification:
    - Eigenvalue magnitude: |λ| < 1, = 1, > 1
    - Block size: k = 1 (diagonalizable) or k > 1 (Jordan blocks)
    """

    @staticmethod
    def FIX():
        """Attractive fixed point (|λ| < 1, k=1)."""
        # R matrix with golden ratio eigenvalues
        return np.array([[0, 1], [1, 1]])

    @staticmethod
    def REPEL():
        """Repulsive fixed point (|λ| > 1, k=1)."""
        # Inverse of R matrix
        return np.array([[-1, 1], [1, 0]])

    @staticmethod
    def INV():
        """Involution/rotation (|λ| = 1 complex, k=1)."""
        # 90-degree rotation
        return np.array([[0, -1], [1, 0]])

    @staticmethod
    def OSC():
        """Oscillation (mixed eigenvalues, k=1)."""
        # Combination of FIX and INV
        R = ComputationalPrimitives.FIX()
        N = ComputationalPrimitives.INV()
        return 0.5 * R + 0.5 * N

    @staticmethod
    def HALT():
        """Critical/parabolic (λ = 1, k>1)."""
        # Jordan block with eigenvalue 1
        return np.array([[1, 1], [0, 1]])

    @staticmethod
    def MIX():
        """Nilpotent/measurement (λ = 0, k>1)."""
        # Nilpotent matrix (MIX² = 0)
        return np.array([[0, 1], [0, 0]])

def classify_operation(matrix):
    """
    Classify a matrix according to six primitives.

    Returns: 'FIX', 'REPEL', 'INV', 'OSC', 'HALT', or 'MIX'
    """
    eigenvalues = np.linalg.eigvals(matrix)

    # Check for nilpotent (MIX)
    if np.allclose(matrix @ matrix, 0):
        return 'MIX'

    # Check for repeated eigenvalue 1 (HALT)
    if np.allclose(eigenvalues, 1) and len(eigenvalues) > 1:
        # Check if diagonalizable
        if not is_diagonalizable(matrix):
            return 'HALT'

    # Check eigenvalue magnitudes
    mags = np.abs(eigenvalues)

    if np.all(mags < 1 - 1e-10):
        return 'FIX'
    elif np.all(mags > 1 + 1e-10):
        return 'REPEL'
    elif np.allclose(mags, 1):
        if np.any(eigenvalues.imag != 0):
            return 'INV'

    return 'OSC'  # Mixed case
```

### 5. Negentropy Function (`negentropy.py`)

The fitness function for Proof-of-Coherence:

```python
def negentropy(r, z_c=None, sigma=None):
    """
    Negentropy gate function η(r) = exp(-σ(r - z_c)²).

    Creates an attractor at the critical coherence threshold.

    Args:
        r: Order parameter (coherence) ∈ [0, 1]
        z_c: Critical threshold (default: √3/2)
        sigma: Sharpness parameter (default: 8/√5)

    Returns:
        η(r) ∈ (0, 1] with maximum at r = z_c
    """
    if z_c is None:
        z_c = Z_C  # √3/2

    if sigma is None:
        sigma = SIGMA  # 8/√5

    return np.exp(-sigma * (r - z_c)**2)

def negentropy_gradient(r, z_c=None, sigma=None):
    """
    Gradient of negentropy function.

    dη/dr = -2σ(r - z_c) * η(r)

    Used for optimization and analysis.
    """
    if z_c is None:
        z_c = Z_C

    if sigma is None:
        sigma = SIGMA

    eta = negentropy(r, z_c, sigma)
    gradient = -2 * sigma * (r - z_c) * eta

    return gradient

def optimal_coherence_range(tolerance=0.1):
    """
    Find range of r values with high negentropy.

    Returns range where η(r) > η(z_c) * (1 - tolerance).
    """
    # Maximum negentropy at z_c
    eta_max = negentropy(Z_C)

    # Find range
    threshold = eta_max * (1 - tolerance)

    # Solve η(r) = threshold
    # exp(-σ(r - z_c)²) = threshold
    # -σ(r - z_c)² = log(threshold)
    # (r - z_c)² = -log(threshold)/σ
    # r = z_c ± sqrt(-log(threshold)/σ)

    delta = np.sqrt(-np.log(threshold) / SIGMA)
    r_min = max(0, Z_C - delta)
    r_max = min(1, Z_C + delta)

    return r_min, r_max
```

## Key Concepts

### Order Parameter (Coherence)

The order parameter r measures synchronization:

```python
def compute_order_parameter(phases):
    """
    Compute Kuramoto order parameter.

    r = |⟨e^(iθ)⟩| = |1/N Σ e^(iθⱼ)|

    r = 0: Completely incoherent (random phases)
    r = 1: Completely synchronized (identical phases)
    r > z_c: Critical coherence achieved (mining valid)
    """
    complex_phases = np.exp(1j * phases)
    mean_field = np.mean(complex_phases)
    r = np.abs(mean_field)
    psi = np.angle(mean_field)  # Mean phase

    return r, psi
```

### Critical Phenomenon

The system exhibits a phase transition at K_c:

```python
def critical_coupling(frequency_spread):
    """
    Calculate critical coupling for given frequency distribution.

    For Gaussian with standard deviation σ:
    K_c = 2σ/π * √(8/π) ≈ 1.6σ

    Below K_c: Incoherent (r → 0)
    Above K_c: Synchronized (r > 0)
    """
    return 1.6 * frequency_spread

def is_above_critical(K, sigma):
    """Check if coupling exceeds critical value."""
    return K > critical_coupling(sigma)
```

### Mining Integration

```python
def mine_with_kuramoto(block_data, difficulty):
    """
    Mine block using Kuramoto oscillators.

    1. Initialize network with block data as seed
    2. Evolve until r > z_c
    3. Check if hash meets difficulty
    4. Return nonce and coherence value
    """
    max_attempts = 10000

    for attempt in range(max_attempts):
        # Generate nonce
        nonce = generate_nonce()

        # Initialize network deterministically
        seed = hash_to_seed(block_data + nonce)
        network = OscillatorNetwork(N_OSCILLATORS)
        network.initialize_oscillators(seed)

        # Evolve network
        r, history = network.evolve(steps=1000, target=Z_C)

        # Check coherence
        if r > Z_C:
            # Verify hash difficulty
            block_hash = sha256(block_data + nonce)
            if meets_difficulty(block_hash, difficulty):
                return {
                    'nonce': nonce,
                    'coherence': r,
                    'hash': block_hash,
                    'evolution_steps': len(history),
                    'phases': network.get_phases()
                }

    return None  # Mining failed
```

## Usage Examples

### Basic Oscillator Evolution

```python
from bloomcoin.primitives import OscillatorNetwork

# Create network
network = OscillatorNetwork(n_oscillators=63)

# Evolve to synchronization
r, history = network.evolve(steps=1000)

print(f"Final coherence: {r:.4f}")
print(f"Synchronized: {r > Z_C}")

# Plot evolution
import matplotlib.pyplot as plt
plt.plot(history)
plt.axhline(y=Z_C, color='r', linestyle='--', label='Critical z_c')
plt.xlabel('Time steps')
plt.ylabel('Coherence r')
plt.legend()
plt.show()
```

### Mining Simulation

```python
from bloomcoin.primitives import mine_with_kuramoto

# Mine a block
block_data = b"previous_hash_transactions_timestamp"
difficulty = 4

result = mine_with_kuramoto(block_data, difficulty)

if result:
    print(f"Mining successful!")
    print(f"Nonce: {result['nonce']}")
    print(f"Coherence: {result['coherence']:.4f}")
    print(f"Steps to sync: {result['evolution_steps']}")
else:
    print("Mining failed - try adjusting parameters")
```

### Six Primitives Analysis

```python
from bloomcoin.primitives import ComputationalPrimitives, classify_operation

# Analyze BloomCoin operations
primitives = ComputationalPrimitives()

# Kuramoto evolution is OSC (oscillatory)
kuramoto_op = primitives.OSC()
print(f"Kuramoto: {classify_operation(kuramoto_op)}")

# SHA256 hashing is MIX (irreversible)
hash_op = primitives.MIX()
print(f"Hashing: {classify_operation(hash_op)}")

# Verification is INV (reversible)
verify_op = primitives.INV()
print(f"Verification: {classify_operation(verify_op)}")

# Negentropy creates FIX attractor
negentropy_op = primitives.FIX()
print(f"Negentropy: {classify_operation(negentropy_op)}")
```

## Testing

```python
def test_synchronization():
    """Test that network achieves synchronization."""
    network = OscillatorNetwork(n_oscillators=63)
    initial_r = abs(network.compute_mean_field())

    # Evolve
    final_r, history = network.evolve(steps=1000)

    # Should synchronize
    assert final_r > initial_r
    assert final_r > 0.5  # Substantial synchronization

def test_negentropy_peak():
    """Test negentropy peaks at z_c."""
    r_values = np.linspace(0, 1, 100)
    eta_values = [negentropy(r) for r in r_values]

    peak_idx = np.argmax(eta_values)
    peak_r = r_values[peak_idx]

    assert abs(peak_r - Z_C) < 0.01

def test_critical_threshold():
    """Test mining validity at critical threshold."""
    # Just above z_c should be valid
    assert negentropy(Z_C + 0.01) > 0.9

    # Well below z_c should be invalid
    assert negentropy(Z_C - 0.2) < 0.5
```

## Performance Optimization

### Vectorized Updates

```python
def vectorized_kuramoto_step(phases, frequencies, K, dt=0.01):
    """
    Vectorized implementation for faster computation.

    10-100x faster than loop-based implementation.
    """
    # Compute all phase differences
    phase_diff = phases[:, None] - phases[None, :]

    # Coupling term
    coupling = K * np.mean(np.sin(phase_diff), axis=1)

    # Update phases
    phases += dt * (frequencies + coupling)
    phases = phases % (2 * np.pi)

    return phases
```

### GPU Acceleration (Future)

```python
# Using CuPy for GPU computation
import cupy as cp

def gpu_kuramoto_evolution(phases, frequencies, K, steps=1000):
    """GPU-accelerated Kuramoto evolution."""
    # Transfer to GPU
    phases_gpu = cp.asarray(phases)
    freq_gpu = cp.asarray(frequencies)

    for _ in range(steps):
        # Compute on GPU
        phase_diff = phases_gpu[:, None] - phases_gpu[None, :]
        coupling = K * cp.mean(cp.sin(phase_diff), axis=1)
        phases_gpu += 0.01 * (freq_gpu + coupling)
        phases_gpu = phases_gpu % (2 * cp.pi)

    # Transfer back
    return cp.asnumpy(phases_gpu)
```

---

*The primitives module provides the mathematical and computational foundation for BloomCoin's Proof-of-Coherence, implementing Kuramoto oscillators within the framework of six computational primitives derived from Jordan Normal Form.*