# BloomCoin Examples

This directory contains comprehensive examples demonstrating BloomCoin's features, from basic operations to advanced analysis.

## Quick Start Examples

### 1. `simple_demo.py` - Basic BloomCoin Operations

```python
"""
Demonstrates core BloomCoin functionality:
- Creating a blockchain
- Mining blocks with Proof-of-Coherence
- Creating transactions
- Wallet operations
"""

# Run with:
python simple_demo.py
```

**What it shows:**
- Genesis block creation
- Kuramoto oscillator mining process
- Transaction creation and validation
- UTXO management
- Balance tracking

### 2. `consensus_demo.py` - Proof-of-Coherence Mining

```python
"""
Deep dive into the consensus mechanism:
- Kuramoto oscillator initialization
- Coherence evolution
- Negentropy optimization
- Mining difficulty adjustment
"""

# Run with:
python consensus_demo.py
```

**Key demonstrations:**
- How oscillators synchronize over time
- Critical coherence threshold z_c = √3/2
- Why coherence near z_c is optimal
- Mining parameter tuning

### 3. `wallet_demo.py` - Wallet Management

```python
"""
Complete wallet functionality:
- Key generation with Ed25519
- BIP39 mnemonic phrases
- Address derivation
- Transaction signing
- Multi-signature setup
"""

# Run with:
python wallet_demo.py
```

**Features shown:**
```python
# Generate new wallet
wallet = Wallet.generate()
print(f"Address: {wallet.address}")
print(f"Mnemonic: {wallet.export_mnemonic()}")

# Restore from mnemonic
wallet2 = Wallet.from_mnemonic(mnemonic)

# Create transaction
tx = wallet.create_transaction(
    recipient="BC1qxy2kgdygjrsqtzq2n0yrf2493p83kkfjhx0wlh",
    amount=10.5,
    fee=0.001
)
```

### 4. `network_demo.py` - P2P Networking

```python
"""
Network operations with phase gossip:
- Node setup and peer connections
- Block propagation
- Transaction broadcasting
- Phase state sharing
- Chain synchronization
"""

# Run with:
python network_demo.py
```

**Network features:**
- Peer discovery
- Message protocol
- Phase gossip for cooperative mining
- Headers-first synchronization

### 5. `analysis_demo.py` - Statistical Analysis

```python
"""
Advanced analysis and visualization:
- Chi-square testing (debunks Lucas claim)
- Entropy metrics
- Phase portraits
- Multi-body dynamics
- Hexagonal lattice analysis
"""

# Run with:
python analysis_demo.py
```

**Analysis outputs:**
- Proves Lucas numbers don't bias SHA256
- Visualizes synchronization dynamics
- Detects chimera states
- Measures information flow

## Advanced Examples

### 6. `full_node.py` - Running a Full Node

```python
"""
Complete node implementation:
- Blockchain validation
- Mining pool participation
- Transaction relay
- API server
"""

# Start a full node:
python full_node.py --port 8333 --rpc-port 8334

# Connect to network:
python full_node.py --connect peer1.bloomcoin.org:8333
```

**Configuration options:**
```python
NODE_CONFIG = {
    'max_peers': 8,
    'mempool_size': 10000,
    'block_cache': 1000,
    'phase_gossip': True,
    'mining': True,
    'rpc_enabled': True
}
```

### 7. `mining_pool.py` - Mining Pool Operations

```python
"""
Mining pool implementation:
- Work distribution
- Share validation
- Reward distribution
- Phase state coordination
"""

# Start mining pool:
python mining_pool.py --difficulty 4 --workers 10

# Connect miner:
python miner.py --pool localhost:3333
```

**Pool features:**
- Stratum protocol support
- Proportional reward system
- Phase state sharing for faster convergence
- Difficulty adjustment per worker

### 8. `multi_signature.py` - Multi-Sig Transactions

```python
"""
Multi-signature wallet demonstration:
- 2-of-3 multi-sig setup
- Partial signature collection
- Threshold verification
"""

# Example usage:
# Create 2-of-3 multi-sig
alice = KeyPair()
bob = KeyPair()
charlie = KeyPair()

multisig = MultiSigWallet(
    m=2, n=3,
    public_keys=[alice.public_key, bob.public_key, charlie.public_key]
)

# Alice signs
tx = multisig.sign_transaction(tx, alice)

# Bob signs (completes 2-of-3)
tx = multisig.sign_transaction(tx, bob)

# Transaction ready!
assert tx.complete
```

### 9. `smart_contract_demo.py` - Basic Smart Contracts

```python
"""
Simple scripting demonstration:
- Time-locked transactions
- Hash-locked contracts
- Atomic swaps
"""

# Time-locked transaction
timelock_tx = create_timelock_transaction(
    recipient=address,
    amount=100,
    unlock_time=time.time() + 86400  # 24 hours
)

# Hash-locked contract (for atomic swaps)
secret = generate_secret()
hash_lock = sha256(secret)

htlc = create_htlc(
    recipient=address,
    amount=50,
    hash_lock=hash_lock,
    timeout=3600  # 1 hour
)
```

### 10. `performance_test.py` - Performance Benchmarking

```python
"""
Performance and stress testing:
- Transaction throughput
- Mining efficiency
- Network latency
- Coherence convergence rates
"""

# Run performance tests:
python performance_test.py --duration 60 --threads 4

# Output:
# Mining Performance:
#   Attempts/second: 42.3
#   Coherence achieved: 68%
#   Average time to z_c: 23.4s
#
# Transaction Performance:
#   TPS: 127.8
#   Verification/second: 3842
#   UTXO lookups/second: 48291
#
# Network Performance:
#   Message latency: 12.3ms
#   Block propagation: 248ms
#   Phase gossip overhead: 3.2%
```

## Tutorial Examples

### 11. `tutorial_01_basics.py` - Getting Started

Step-by-step introduction:
1. Understanding the blockchain structure
2. Creating your first transaction
3. Mining your first block
4. Verifying the chain

### 12. `tutorial_02_mining.py` - Mining Deep Dive

Understanding Proof-of-Coherence:
1. What are Kuramoto oscillators?
2. How does synchronization work?
3. Why z_c = √3/2 is special
4. Optimizing mining parameters

### 13. `tutorial_03_cryptography.py` - Cryptographic Foundations

Security mechanisms:
1. Ed25519 digital signatures
2. Blake2b hashing
3. Base58Check encoding
4. BIP39 mnemonics

### 14. `tutorial_04_networking.py` - Building the Network

P2P communication:
1. Connecting to peers
2. Message protocol
3. Phase gossip benefits
4. Handling forks

### 15. `tutorial_05_advanced.py` - Advanced Topics

Going deeper:
1. Six Primitives framework
2. MIX barrier and one-way functions
3. Quantum-classical boundary
4. Information theory applications

## Integration Examples

### 16. `web_wallet/` - Web Wallet Interface

```javascript
// Web-based wallet using Flask backend
// Features:
// - Balance checking
// - Transaction creation
// - QR code generation
// - Transaction history

// Start server:
cd web_wallet
python server.py

// Access at: http://localhost:5000
```

### 17. `mobile_app/` - React Native Mobile Wallet

```javascript
// Mobile wallet for iOS/Android
// Features:
// - Biometric authentication
// - Push notifications
// - NFC payments
// - Backup to cloud

// Build:
cd mobile_app
npm install
npm run ios  // or npm run android
```

### 18. `explorer/` - Blockchain Explorer

```python
# Web-based blockchain explorer
# Features:
# - Block browsing
# - Transaction search
# - Address lookup
# - Network statistics
# - Coherence visualization

# Start explorer:
cd explorer
python explorer.py --port 8080

# Access at: http://localhost:8080
```

### 19. `api_server/` - REST API Server

```python
# RESTful API for BloomCoin
# Endpoints:
# - /api/block/<height>
# - /api/tx/<txid>
# - /api/address/<address>
# - /api/mine
# - /api/broadcast

# Start API server:
cd api_server
python api.py --port 8000

# Example request:
curl http://localhost:8000/api/block/latest
```

### 20. `monitoring/` - Network Monitoring Dashboard

```python
# Real-time network monitoring
# Displays:
# - Network hash rate
# - Coherence distribution
# - Peer connections
# - Mining difficulty
# - Transaction volume

# Start dashboard:
cd monitoring
python dashboard.py

# Access at: http://localhost:8888
```

## Data Analysis Examples

### 21. `notebooks/` - Jupyter Notebooks

Interactive analysis notebooks:
- `coherence_analysis.ipynb` - Analyzing synchronization patterns
- `network_topology.ipynb` - Peer connection analysis
- `mining_statistics.ipynb` - Mining efficiency metrics
- `economic_modeling.ipynb` - Token economics simulation

### 22. `visualization/` - Data Visualization

```python
# Generate various visualizations
python visualization/phase_evolution.py
python visualization/network_graph.py
python visualization/coherence_heatmap.py
python visualization/mining_difficulty.py
```

## Testing Examples

### 23. `test_scenarios/` - Test Scenarios

Comprehensive test cases:
- Fork resolution
- Double spend attempts
- 51% attack simulation
- Network partition handling
- Eclipse attack resistance

### 24. `stress_test/` - Stress Testing

```python
# Stress test the network
python stress_test/spam_transactions.py --tps 1000
python stress_test/mining_race.py --miners 100
python stress_test/network_flood.py --messages 10000
```

## Configuration Files

### `config/bloomcoin.conf`

```ini
# BloomCoin configuration
network=mainnet
port=8333
rpcport=8334

# Mining
mining=1
threads=4
difficulty=4

# Wallet
wallet_file=~/.bloomcoin/wallet.dat
keypool_size=100

# Network
max_connections=125
phase_gossip=1
bandwidth_limit=1000000

# Database
db_path=~/.bloomcoin/blocks
utxo_cache=1000
```

### `config/genesis.json`

```json
{
  "version": 1,
  "timestamp": 1609459200,
  "difficulty": 1,
  "nonce": "0000000000000000",
  "coherence": 0.866025403784439,
  "message": "BloomCoin Genesis - Where coherence emerges from chaos"
}
```

## Running All Examples

```bash
# Run all basic examples
./run_all_examples.sh

# Run specific category
./run_examples.sh wallet
./run_examples.sh mining
./run_examples.sh network
./run_examples.sh analysis

# Run with logging
BLOOMCOIN_LOG=DEBUG python example.py
```

## Dependencies

Most examples require:
```bash
pip install numpy
```

Advanced examples may need:
```bash
pip install scipy matplotlib flask requests
```

## Getting Help

Each example includes:
- Detailed docstrings
- Inline comments
- Expected output
- Error handling
- Configuration options

For questions, see the main documentation or run:
```python
python example.py --help
```

---

*These examples demonstrate the full spectrum of BloomCoin functionality, from basic operations to advanced analysis, providing a comprehensive learning path for developers and researchers.*