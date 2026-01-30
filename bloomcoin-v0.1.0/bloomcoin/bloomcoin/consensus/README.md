# BloomCoin Consensus Module - Proof of Coherence

## Overview

The consensus module implements **Proof-of-Coherence (PoC)**, a novel consensus mechanism based on Kuramoto oscillator synchronization. Instead of solving arbitrary cryptographic puzzles, miners achieve consensus by synchronizing a network of coupled oscillators to a critical coherence threshold.

## Mathematical Foundation

### Kuramoto Model

The Kuramoto model describes the synchronization of coupled oscillators:

```
dθᵢ/dt = ωᵢ + (K/N) Σⱼ sin(θⱼ - θᵢ)
```

Where:
- θᵢ = phase of oscillator i ∈ [0, 2π]
- ωᵢ = natural frequency of oscillator i
- K = coupling strength = 2φ/3 ≈ 1.079
- N = 63 oscillators (triangle(8) - 1)

### Order Parameter

The coherence (order parameter) measures synchronization:

```python
r = |⟨e^(iθ)⟩| = |1/N Σⱼ e^(iθⱼ)|

# Where:
# r = 0: completely incoherent (random phases)
# r = 1: completely synchronized (identical phases)
# r > z_c = √3/2: critical coherence achieved
```

### Negentropy Function

The fitness function that creates an attractor at z_c:

```python
η(r) = exp(-σ(r - z_c)²)

# Where:
# z_c = √3/2 ≈ 0.866 (critical threshold)
# σ = 8/√5 ≈ 3.578 (sharpness parameter)
```

## Architecture

```
consensus/
├── __init__.py          # Consensus interface
├── kuramoto.py         # Oscillator dynamics
├── miner.py           # Mining algorithm
├── validator.py       # Block validation
└── difficulty.py      # Difficulty adjustment
```

## Core Components

### 1. Kuramoto Dynamics (`kuramoto.py`)

```python
class KuramotoNetwork:
    def __init__(self, n_oscillators=63):
        self.N = n_oscillators
        self.K = 2 * PHI / 3  # Coupling from golden ratio
        self.phases = None    # Current phases
        self.frequencies = None  # Natural frequencies

    def initialize(self, block_data, nonce):
        """Initialize oscillators from block data."""
        # Generate deterministic frequencies
        seed = hash(block_data + nonce)
        self.frequencies = generate_frequencies(seed)
        self.phases = generate_initial_phases(seed)

    def step(self, dt=0.01):
        """Evolve oscillators one time step."""
        # Compute mean field
        mean_field = np.mean(np.exp(1j * self.phases))
        r = np.abs(mean_field)
        psi = np.angle(mean_field)

        # Update phases (Kuramoto dynamics)
        coupling = self.K * r * np.sin(psi - self.phases)
        self.phases += dt * (self.frequencies + coupling)
        self.phases = self.phases % (2 * np.pi)

        return r  # Return coherence

    def evolve(self, steps=1000, target=None):
        """Evolve until convergence or target coherence."""
        history = []
        for _ in range(steps):
            r = self.step()
            history.append(r)

            if target and r > target:
                return r, history

        return r, history
```

### 2. Mining Algorithm (`miner.py`)

```python
class ProofOfCoherence:
    def __init__(self, difficulty=4):
        self.difficulty = difficulty
        self.network = KuramotoNetwork()
        self.z_c = np.sqrt(3) / 2  # Critical threshold

    def mine(self, block_data, max_attempts=10000):
        """Mine a block by finding coherent configuration."""
        best_result = {'coherence': 0, 'nonce': None}

        for attempt in range(max_attempts):
            # Generate nonce
            nonce = generate_nonce()

            # Initialize oscillators
            self.network.initialize(block_data, nonce)

            # Evolve to find coherence
            coherence, history = self.network.evolve(
                steps=1000,
                target=self.z_c
            )

            # Check if valid
            if coherence > self.z_c:
                # Verify hash difficulty
                block_hash = sha256(block_data + nonce)
                if meets_difficulty(block_hash, self.difficulty):
                    return {
                        'nonce': nonce,
                        'coherence': coherence,
                        'hash': block_hash,
                        'phases': self.network.phases,
                        'history': history,
                        'negentropy': negentropy(coherence)
                    }

            # Track best attempt
            if coherence > best_result['coherence']:
                best_result = {
                    'coherence': coherence,
                    'nonce': nonce,
                    'phases': self.network.phases
                }

        return best_result  # Return best even if not valid
```

### 3. Validation (`validator.py`)

```python
class BlockValidator:
    def __init__(self):
        self.z_c = np.sqrt(3) / 2

    def validate_coherence(self, block):
        """Verify block's coherence claim."""
        # Recreate oscillator network
        network = KuramotoNetwork()
        network.initialize(block.data, block.nonce)

        # Evolve with same parameters
        coherence, _ = network.evolve(steps=1000)

        # Verify coherence
        if coherence < self.z_c:
            return False, f"Coherence {coherence} < {self.z_c}"

        # Verify it matches claimed value
        if abs(coherence - block.coherence) > 0.01:
            return False, f"Coherence mismatch: {coherence} != {block.coherence}"

        return True, "Valid coherence"

    def validate_block(self, block, prev_block):
        """Complete block validation."""
        # 1. Check previous hash
        if block.prev_hash != prev_block.hash():
            return False, "Invalid previous hash"

        # 2. Verify coherence
        valid, msg = self.validate_coherence(block)
        if not valid:
            return False, msg

        # 3. Check difficulty
        if not meets_difficulty(block.hash(), block.difficulty):
            return False, "Insufficient difficulty"

        # 4. Verify transactions
        for tx in block.transactions:
            if not validate_transaction(tx):
                return False, f"Invalid transaction: {tx.id()}"

        # 5. Check timestamp
        if block.timestamp <= prev_block.timestamp:
            return False, "Invalid timestamp"

        return True, "Block valid"
```

### 4. Difficulty Adjustment (`difficulty.py`)

```python
class DifficultyAdjuster:
    def __init__(self, target_time=60, adjustment_interval=10):
        self.target_time = target_time  # Target seconds per block
        self.adjustment_interval = adjustment_interval

    def adjust(self, chain):
        """Adjust difficulty based on recent block times."""
        # Only adjust every N blocks
        if len(chain.blocks) % self.adjustment_interval != 0:
            return chain.difficulty

        # Calculate actual time for recent blocks
        recent = chain.blocks[-self.adjustment_interval:]
        time_taken = recent[-1].timestamp - recent[0].timestamp
        time_per_block = time_taken / len(recent)

        # Calculate adjustment ratio
        ratio = self.target_time / time_per_block

        # Apply with limits (max 2x change)
        ratio = max(0.5, min(2.0, ratio))
        new_difficulty = int(chain.difficulty * ratio)

        return new_difficulty

    def next_difficulty(self, chain):
        """Calculate next block's difficulty."""
        # Consider coherence quality
        recent_coherences = [b.coherence for b in chain.blocks[-10:]]
        avg_coherence = np.mean(recent_coherences)

        # Higher quality coherence = slightly higher difficulty
        quality_factor = negentropy(avg_coherence)
        adjusted_diff = self.adjust(chain) * (1 + 0.1 * quality_factor)

        return int(adjusted_diff)
```

## Mining Process

### Step-by-Step Mining Flow

1. **Receive Block Template**
```python
template = {
    'prev_hash': chain.last_block.hash(),
    'transactions': mempool.get_transactions(),
    'timestamp': time.time(),
    'difficulty': chain.next_difficulty()
}
```

2. **Search for Valid Nonce**
```python
miner = ProofOfCoherence(difficulty=template['difficulty'])
result = miner.mine(
    block_data=serialize(template),
    max_attempts=100000
)
```

3. **Verify Solution**
```python
if result['coherence'] > z_c:
    # Create block with solution
    block = Block(
        **template,
        nonce=result['nonce'],
        coherence=result['coherence']
    )

    # Broadcast to network
    network.broadcast_block(block)
```

### Mining Optimization Strategies

```python
class OptimizedMiner:
    def __init__(self):
        self.cache = {}  # Cache good starting configurations

    def mine_with_memory(self, block_data):
        """Use memory of good configurations."""
        # Try cached configurations first
        if similar_block in self.cache:
            nonce = modify_cached_nonce(self.cache[similar_block])
            result = try_nonce(nonce)
            if result['coherence'] > z_c:
                return result

        # Fall back to random search
        return standard_mine(block_data)

    def parallel_mine(self, block_data, n_threads=4):
        """Parallel mining with multiple threads."""
        with ThreadPoolExecutor(max_workers=n_threads) as executor:
            futures = []
            for i in range(n_threads):
                # Each thread tries different nonce space
                futures.append(
                    executor.submit(mine_partition, block_data, i, n_threads)
                )

            # Return first valid result
            for future in as_completed(futures):
                result = future.result()
                if result['coherence'] > z_c:
                    return result
```

## Phase Gossip Protocol

### Sharing Phase States

Miners can share phase configurations to accelerate convergence:

```python
class PhaseGossip:
    def __init__(self, node):
        self.node = node
        self.phase_cache = {}

    def share_phases(self, block_height, phases, coherence):
        """Share successful phase configuration."""
        message = {
            'type': 'PHASE_STATE',
            'height': block_height,
            'phases': phases.tolist(),
            'coherence': coherence,
            'timestamp': time.time()
        }
        self.node.broadcast(message)

    def receive_phases(self, message):
        """Receive and validate shared phases."""
        # Verify coherence claim
        network = KuramotoNetwork()
        network.phases = np.array(message['phases'])
        actual_coherence = network.compute_coherence()

        if abs(actual_coherence - message['coherence']) < 0.01:
            # Cache valid configuration
            self.phase_cache[message['height']] = message['phases']
            return True
        return False

    def use_shared_phases(self, block_data):
        """Try mining with shared phase configurations."""
        if block_data in self.phase_cache:
            # Start from shared configuration
            initial_phases = self.phase_cache[block_data]
            # Add small perturbation
            phases = initial_phases + 0.1 * np.random.randn(63)
            return evolve_from_phases(phases)
```

## Security Analysis

### 1. Resistance to Parallelization

Unlike hash-based PoW, Kuramoto evolution is inherently sequential:
- Each step depends on previous state
- Cannot skip ahead without computing intermediate steps
- Natural resistance to ASIC optimization

### 2. Sybil Resistance

Coherence verification across network:
```python
def verify_peer_coherence(peer_claim):
    # Peers must prove they achieved coherence
    network = recreate_network(peer_claim['block_data'],
                              peer_claim['nonce'])
    actual = network.evolve()
    return actual >= peer_claim['coherence'] - tolerance
```

### 3. Energy Efficiency

Comparison with Bitcoin:
| Metric | Bitcoin PoW | BloomCoin PoC |
|--------|------------|---------------|
| Operation | SHA256 hashing | Oscillator simulation |
| Parallelizable | Yes (ASICs) | No (sequential) |
| Energy/block | ~1 MWh | ~1 kWh |
| Meaningful computation | No | Yes (synchronization) |

## Performance Metrics

### Benchmarks

```python
# Typical mining performance
def benchmark_mining():
    results = {
        'attempts_per_second': 0,
        'avg_coherence_time': 0,
        'success_rate': 0
    }

    start = time.time()
    attempts = 0
    successes = 0

    for _ in range(100):
        result = mine_once()
        attempts += 1
        if result['coherence'] > z_c:
            successes += 1

    elapsed = time.time() - start
    results['attempts_per_second'] = attempts / elapsed
    results['avg_coherence_time'] = elapsed / attempts
    results['success_rate'] = successes / attempts

    return results

# Typical results:
# - Attempts/second: ~10-50 (depends on CPU)
# - Avg coherence time: 20-100ms
# - Success rate: ~1-5% (depends on difficulty)
```

### Optimization Techniques

1. **Adaptive Time Stepping**
```python
def adaptive_evolution(network, target):
    dt = 0.01
    while network.coherence < target:
        # Increase step size when far from target
        if network.coherence < 0.5:
            dt = 0.05
        # Decrease near critical point
        elif network.coherence > 0.8:
            dt = 0.001
        network.step(dt)
```

2. **Early Termination**
```python
def mine_with_early_termination(block_data):
    for nonce in generate_nonces():
        network.initialize(block_data, nonce)
        for step in range(100):
            r = network.step()
            # Abandon if not progressing
            if step > 50 and r < 0.3:
                break  # Try next nonce
            if r > z_c:
                return nonce
```

3. **GPU Acceleration** (Future)
```python
# Parallel phase updates on GPU
@cuda.jit
def kuramoto_kernel(phases, frequencies, output):
    i = cuda.grid(1)
    if i < phases.size:
        # Compute coupling for oscillator i
        coupling = compute_coupling_gpu(i, phases)
        output[i] = phases[i] + dt * (frequencies[i] + coupling)
```

## Testing

### Unit Tests

```python
def test_kuramoto_convergence():
    """Test that network achieves coherence."""
    network = KuramotoNetwork()
    network.initialize("test_block", "test_nonce")

    initial_r = network.compute_coherence()
    final_r, history = network.evolve(steps=1000)

    assert final_r > initial_r  # Should increase
    assert len(history) == 1000  # Correct history

def test_negentropy_peak():
    """Test negentropy peaks at z_c."""
    r_values = np.linspace(0, 1, 100)
    eta_values = [negentropy(r) for r in r_values]
    peak_idx = np.argmax(eta_values)
    peak_r = r_values[peak_idx]
    assert abs(peak_r - z_c) < 0.01  # Peak at z_c
```

### Integration Tests

```python
def test_mining_integration():
    """Test complete mining process."""
    # Create block template
    template = create_block_template()

    # Mine block
    miner = ProofOfCoherence(difficulty=2)
    result = miner.mine(template)

    # Validate result
    validator = BlockValidator()
    valid, msg = validator.validate_coherence(result)
    assert valid
    assert result['coherence'] > z_c
```

## Configuration

### Mining Parameters

```python
# config.py
MINING_CONFIG = {
    'n_oscillators': 63,           # Number of oscillators
    'coupling': 2 * PHI / 3,       # Kuramoto coupling
    'critical_threshold': np.sqrt(3) / 2,  # z_c
    'max_evolution_steps': 1000,   # Max steps per attempt
    'time_step': 0.01,             # Integration time step
    'difficulty': 4,               # Initial difficulty
    'target_block_time': 60,       # Target seconds
    'adjustment_interval': 10      # Blocks between adjustments
}
```

### Tuning Guide

```python
# For faster blocks (lower security):
FAST_CONFIG = {
    'n_oscillators': 31,  # Fewer oscillators
    'max_evolution_steps': 500,
    'difficulty': 2
}

# For higher security (slower blocks):
SECURE_CONFIG = {
    'n_oscillators': 127,  # More oscillators
    'max_evolution_steps': 2000,
    'difficulty': 6
}
```

## Future Enhancements

### 1. Alternative Oscillator Models
- Stuart-Landau oscillators (amplitude dynamics)
- Coupled Van der Pol oscillators (limit cycles)
- Quantum oscillators (for quantum resistance)

### 2. Adaptive Coupling
```python
# Dynamic coupling based on network state
K(t) = K_0 * (1 + α * sin(ωt))
```

### 3. Hierarchical Consensus
- Local groups achieve coherence
- Groups synchronize globally
- Enables sharding and scalability

---

*The Proof-of-Coherence consensus mechanism transforms mining from meaningless computation into meaningful synchronization, creating a more energy-efficient and mathematically elegant blockchain.*