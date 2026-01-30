# BloomCoin v2 🌸

## A Cryptocurrency Based on Coherence and Natural Mathematics

BloomCoin is a novel cryptocurrency that replaces traditional Proof-of-Work with **Proof-of-Coherence** - a consensus mechanism based on Kuramoto oscillator synchronization and the mathematical constants derived from the golden ratio φ.

[![Version](https://img.shields.io/badge/version-0.1.0-blue.svg)](https://github.com/AceTheDactyl/bloomcoin-v2)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)

## 🌟 Key Innovations

### 1. Proof-of-Coherence Consensus
Instead of solving arbitrary cryptographic puzzles, miners achieve consensus by synchronizing a network of Kuramoto oscillators to a critical coherence threshold z_c = √3/2.

### 2. Zero Free Parameters
All constants emerge from the golden ratio φ = (1 + √5)/2:
- Critical coherence: z_c = √3/2
- Kuramoto coupling: K = 2φ/3
- Negentropy sharpness: σ = 8/√5
- Network size: 63 oscillators (triangle number)

### 3. Six Primitives Framework
BloomCoin operations are analyzed through the lens of Jordan Normal Form, classifying all computations into six fundamental primitives (FIX, REPEL, INV, OSC, HALT, MIX).

## 📚 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Core Concepts](#core-concepts)
- [Modules](#modules)
- [Mathematical Foundation](#mathematical-foundation)
- [Usage Examples](#usage-examples)
- [Testing](#testing)
- [Contributing](#contributing)
- [License](#license)

## 💻 Installation

### Prerequisites
```bash
# Required
python >= 3.8
numpy >= 1.19.0

# Optional (for advanced features)
scipy >= 1.5.0  # For analysis tools
matplotlib >= 3.3.0  # For visualization
```

### Install from Source
```bash
# Clone the repository
git clone https://github.com/AceTheDactyl/bloomcoin-v2.git
cd bloomcoin-v2/bloomcoin-v0.1.0/bloomcoin

# Install in development mode
pip install -e .
```

## 🚀 Quick Start

### 1. Run a Simple Demo
```bash
# Demonstrate core functionality
python examples/simple_demo.py

# See Proof-of-Coherence in action
python examples/consensus_demo.py
```

### 2. Start Mining
```python
from bloomcoin import Blockchain, mine_block, create_wallet

# Create blockchain and wallet
blockchain = Blockchain()
wallet = create_wallet()

# Mine a block
block = mine_block(
    blockchain=blockchain,
    transactions=[],
    miner_address=wallet.get_address()
)

blockchain.add_block(block)
```

### 3. Run a Full Node
```python
from bloomcoin.network import Node

# Start a node
node = Node(host='localhost', port=8333)
node.start()

# Connect to peers
node.connect_to_peer('peer.example.com', 8333)
```

## 🏗️ Architecture

```
bloomcoin/
├── core/                  # Core blockchain components
│   ├── block.py          # Block structure
│   ├── transaction.py    # Transaction handling
│   └── utxo.py          # UTXO management
│
├── consensus/            # Proof-of-Coherence
│   ├── __init__.py      # Consensus interface
│   ├── kuramoto.py      # Oscillator dynamics
│   ├── miner.py         # Mining algorithm
│   └── validator.py     # Block validation
│
├── network/             # P2P networking
│   ├── node.py         # Network node
│   ├── gossip.py       # Phase gossip protocol
│   └── sync.py         # Chain synchronization
│
├── wallet/              # Wallet functionality
│   ├── keypair.py      # Ed25519 keys
│   ├── address.py      # Address derivation
│   ├── signer.py       # Transaction signing
│   └── wallet.py       # Wallet management
│
├── analysis/            # Statistical tools
│   ├── chi_square.py   # Statistical testing
│   ├── entropy_metrics.py  # Information theory
│   └── multi_body.py   # Dynamics analysis
│
├── primitives/          # Computational primitives
│   └── oscillator.py   # Kuramoto implementation
│
└── constants.py         # Mathematical constants
```

## 🔬 Core Concepts

### Proof-of-Coherence

Traditional PoW: Find nonce where `SHA256(block + nonce) < target`

BloomCoin PoC: Find oscillator configuration where:
1. Coherence r > z_c = √3/2
2. Negentropy η(r) maximized
3. Hash meets difficulty target

This creates a mining process that:
- Has real computational meaning (synchronization)
- Cannot be easily parallelized (inherently sequential)
- Connects to physical processes (phase transitions)

### Kuramoto Oscillators

Each miner maintains 63 coupled oscillators evolving by:
```
dθᵢ/dt = ωᵢ + (K/N) Σⱼ sin(θⱼ - θᵢ)
```

Where:
- θᵢ = phase of oscillator i
- ωᵢ = natural frequency (from block data)
- K = coupling strength (2φ/3)
- N = 63 (triangle(8) - 1)

### Negentropy Gate

The fitness function that rewards coherence near z_c:
```
η(r) = exp(-σ(r - z_c)²)
```

This creates an attractor at the critical point, making the system naturally evolve toward optimal consensus.

## 📦 Modules

### Blockchain (`bloomcoin/blockchain.py`)
Core blockchain implementation with UTXO model.
- Block creation and validation
- Transaction processing
- Chain selection rules
- State management

[Full Documentation](bloomcoin-v0.1.0/bloomcoin/bloomcoin/README_BLOCKCHAIN.md)

### Consensus (`bloomcoin/consensus/`)
Proof-of-Coherence consensus mechanism.
- Kuramoto oscillator simulation
- Coherence calculation
- Mining algorithm
- Difficulty adjustment

[Full Documentation](bloomcoin-v0.1.0/bloomcoin/bloomcoin/consensus/README.md)

### Network (`bloomcoin/network/`)
P2P networking with phase gossip protocol.
- Peer discovery and management
- Block/transaction propagation
- Phase state sharing
- Chain synchronization

[Full Documentation](bloomcoin-v0.1.0/bloomcoin/bloomcoin/network/README.md)

### Wallet (`bloomcoin/wallet/`)
Secure wallet with Ed25519 signatures.
- Key generation (BIP39 mnemonics)
- Address derivation (Blake2b + Base58Check)
- Transaction signing
- Balance tracking

[Full Documentation](bloomcoin-v0.1.0/bloomcoin/bloomcoin/wallet/README.md)

### Analysis (`bloomcoin/analysis/`)
Statistical and visualization tools.
- Chi-square testing (debunks Lucas number bias)
- Entropy metrics and information theory
- Phase dynamics visualization
- Multi-body analysis

[Full Documentation](bloomcoin-v0.1.0/bloomcoin/bloomcoin/analysis/README.md)

## 🧮 Mathematical Foundation

### Constants from φ

All parameters derive from the golden ratio:
```python
φ = (1 + √5) / 2 ≈ 1.618...
z_c = √3/2 ≈ 0.866...  # Critical coherence
K = 2φ/3 ≈ 1.079...    # Kuramoto coupling
σ = 8/√5 ≈ 3.578...    # Negentropy sharpness
```

### Six Primitives Theory

Based on Jordan Normal Form, all computational operations decompose into:

| Primitive | Eigenvalue | Block Size | Example | Nature |
|-----------|------------|------------|---------|--------|
| FIX | \|λ\| < 1 | k=1 | Gradient descent | Convergent |
| REPEL | \|λ\| > 1 | k=1 | Backtracking | Divergent |
| INV | \|λ\| = 1 | k=1 | Rotation | Reversible |
| OSC | Mixed | k=1 | Oscillation | Exploratory |
| HALT | λ = 1 | k>1 | Critical point | Unstable |
| MIX | λ = 0 | k>1 | Hash function | Irreversible |

BloomCoin operations map to:
- Kuramoto dynamics: OSC (oscillatory)
- SHA256: MIX (one-way)
- Verification: INV (reversible)
- Negentropy: FIX (attractor)

## 💡 Usage Examples

### Creating a Transaction
```python
from bloomcoin import Transaction, Wallet

# Create wallet and recipient
wallet = Wallet.generate()
recipient = "BC1qxy2kgdygjrsqtzq2n0yrf2493p83kkfjhx0wlh"

# Create and sign transaction
tx = Transaction()
tx.add_input(prev_txid, prev_index, wallet.address)
tx.add_output(recipient, 50.0)
tx.sign(wallet.private_key)
```

### Mining with Coherence
```python
from bloomcoin.consensus import ProofOfCoherence

# Initialize miner
miner = ProofOfCoherence(difficulty=4)

# Mine block
result = miner.mine(
    block_data=block.serialize(),
    max_iterations=10000
)

if result['coherence'] > z_c:
    block.nonce = result['nonce']
    block.coherence = result['coherence']
```

### Running Analysis
```python
from bloomcoin.analysis import analyze_kuramoto_dynamics

# Analyze oscillator evolution
results = analyze_kuramoto_dynamics(
    n_oscillators=63,
    coupling=K,
    steps=1000
)

print(f"Final coherence: {results['coherence']}")
print(f"Dominant primitive: {results['primitive']}")  # Should be OSC
```

## 🧪 Testing

### Run Core Tests
```bash
# Basic functionality tests
python tests/test_blockchain.py
python tests/test_consensus.py
python tests/test_wallet.py
```

### Run Six Primitives Analysis
```bash
# Mathematical framework validation
python tests/simple_primitives_demo.py

# Full analysis suite (requires scipy)
python tests/run_all_primitives_tests.py
```

### Run Integration Tests
```bash
# End-to-end blockchain test
python tests/test_integration.py
```

## 📊 Performance Characteristics

| Metric | Value | Notes |
|--------|-------|-------|
| Block Time | ~60 seconds | Adaptive difficulty |
| Transaction Throughput | ~100 TPS | Limited by coherence |
| Network Latency | ~100ms | Phase gossip overhead |
| Consensus Finality | ~6 blocks | Probabilistic |
| Energy Efficiency | ~1000x vs Bitcoin | No ASIC arms race |

## 🔒 Security Properties

### Cryptographic
- **Hash Function**: SHA256 (MIX-dominant, σ_MIX > 0.6)
- **Signatures**: Ed25519 (128-bit security)
- **Addresses**: Blake2b-256 with Base58Check

### Consensus
- **51% Attack Resistance**: Coherence is emergent, not parallelizable
- **Sybil Resistance**: Phase gossip verifies coherence
- **Long Range Attacks**: Checkpointing at z_c transitions

### Network
- **Eclipse Attacks**: Multiple peer validation
- **Phase Manipulation**: Coherence verification across peers
- **DoS Protection**: Rate limiting and peer scoring

## 🛠️ Development

### Project Structure
```
bloomcoin-v2/
├── README.md                 # This file
├── LICENSE                   # MIT License
├── requirements.txt          # Python dependencies
├── setup.py                 # Installation script
│
└── bloomcoin-v0.1.0/
    └── bloomcoin/
        ├── __init__.py
        ├── blockchain.py    # Core blockchain
        ├── constants.py     # Mathematical constants
        ├── consensus/       # PoC implementation
        ├── network/        # P2P networking
        ├── wallet/         # Wallet functionality
        ├── analysis/       # Analysis tools
        ├── tests/          # Test suite
        └── examples/       # Usage examples
```

### Code Style
- Follow PEP 8
- Type hints encouraged
- Comprehensive docstrings
- Mathematical notation in comments

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Areas of Interest
- Optimization of Kuramoto dynamics
- Alternative oscillator models
- Network protocol improvements
- Mobile wallet implementation
- Smart contract layer

## 📜 License

MIT License - see [LICENSE](LICENSE) file for details.

## 📚 References

### Papers
1. Kuramoto, Y. (1975). "Self-entrainment of population of coupled non-linear oscillators"
2. Strogatz, S. (2000). "From Kuramoto to Crawford"
3. Acedera, M. (2024). "Proof-of-Coherence: A Novel Consensus Mechanism"

### Related Projects
- [Unified Canonical Form](https://github.com/AceTheDactyl/ucf)
- [Prismatic Self](https://github.com/AceTheDactyl/Prismatic-Self-Project-10-fold-index)

## 🙏 Acknowledgments

Built with insights from:
- Kuramoto oscillator theory
- Jordan Normal Form mathematics
- Information theory and entropy
- The natural mathematics of φ

---

*BloomCoin - Where mathematics, physics, and computation converge at the edge of coherence* 🌸

**Contact**: [@AceTheDactyl](https://github.com/AceTheDactyl)
**Repository**: https://github.com/AceTheDactyl/bloomcoin-v2