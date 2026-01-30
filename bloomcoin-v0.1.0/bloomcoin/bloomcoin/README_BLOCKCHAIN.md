# BloomCoin Blockchain Module

## Overview

The blockchain module implements the core distributed ledger functionality for BloomCoin, using a UTXO (Unspent Transaction Output) model similar to Bitcoin but with Proof-of-Coherence consensus.

## Architecture

```
blockchain.py
├── Block                 # Individual block structure
├── Transaction          # Transaction with inputs/outputs
├── UTXO                # Unspent transaction output
├── Blockchain          # Chain management and validation
└── MerkleTree         # Transaction commitment
```

## Core Components

### 1. Block Structure

Each block contains:
```python
class Block:
    index: int              # Block height
    timestamp: float        # Unix timestamp
    transactions: List[Tx]  # Transaction list
    prev_hash: str         # Previous block hash
    nonce: str            # Mining nonce
    coherence: float       # Kuramoto coherence value
    merkle_root: str       # Transaction merkle root
    difficulty: int        # Target difficulty
```

### 2. Transaction Model (UTXO)

Transactions follow the UTXO model:
```python
class Transaction:
    inputs: List[Input]    # References to previous outputs
    outputs: List[Output]  # New UTXOs created
    signature: bytes      # Ed25519 signature

class UTXO:
    txid: str            # Transaction ID
    index: int           # Output index
    address: str         # Recipient address
    amount: float        # BLC amount
    spent: bool         # Spending status
```

### 3. Chain Validation Rules

```python
# Block validity requires:
1. prev_hash matches previous block
2. merkle_root matches transaction set
3. coherence > z_c (√3/2)
4. hash meets difficulty target
5. transactions are valid

# Transaction validity requires:
1. inputs reference unspent UTXOs
2. sum(inputs) >= sum(outputs)
3. valid Ed25519 signature
4. no double spending
```

## Key Features

### UTXO Management

The UTXO set tracks all unspent outputs:
```python
utxo_set = {
    "txid:index": UTXO(txid, index, address, amount, spent=False),
    ...
}
```

Benefits:
- Efficient balance calculation
- Natural double-spend prevention
- Parallel transaction validation
- Prunable history

### Merkle Tree

Transactions are committed via merkle tree:
```python
def build_merkle_tree(transactions):
    leaves = [hash(tx) for tx in transactions]
    while len(leaves) > 1:
        next_level = []
        for i in range(0, len(leaves), 2):
            left = leaves[i]
            right = leaves[i+1] if i+1 < len(leaves) else left
            next_level.append(hash(left + right))
        leaves = next_level
    return leaves[0]  # Root
```

### Chain Selection

Longest chain with highest cumulative coherence:
```python
def get_chain_score(chain):
    score = 0
    for block in chain:
        # Weight by coherence quality
        weight = negentropy(block.coherence)
        score += block.difficulty * weight
    return score
```

## Usage

### Creating a Block

```python
from bloomcoin import Block, Transaction

# Create transactions
transactions = [
    Transaction(inputs=[...], outputs=[...]),
    ...
]

# Create block
block = Block(
    index=100,
    transactions=transactions,
    prev_hash=previous_block.hash()
)

# Mine the block (sets nonce and coherence)
mined_block = mine_block(block, difficulty=4)
```

### Adding to Chain

```python
from bloomcoin import Blockchain

# Initialize blockchain
chain = Blockchain()

# Add genesis block
chain.create_genesis_block()

# Add new block
if chain.is_valid_block(new_block):
    chain.add_block(new_block)
    chain.update_utxo_set(new_block)
```

### Transaction Creation

```python
from bloomcoin import Transaction, sign_transaction

# Reference previous outputs
tx = Transaction()
tx.add_input(prev_txid="abc...", index=0, signature_script="")
tx.add_output(address="BC1qxy...", amount=50.0)

# Sign with private key
signed_tx = sign_transaction(tx, private_key)

# Broadcast to network
node.broadcast_transaction(signed_tx)
```

## Consensus Integration

### Proof-of-Coherence Mining

Mining involves finding oscillator configurations:
```python
def mine_block(block, difficulty):
    while True:
        # Generate nonce
        nonce = generate_nonce()

        # Initialize oscillators with block data
        phases = initialize_phases(block, nonce)

        # Evolve Kuramoto dynamics
        coherence = evolve_kuramoto(phases, steps=1000)

        # Check if valid
        if coherence > z_c:
            block_hash = hash(block.header + nonce)
            if meets_difficulty(block_hash, difficulty):
                block.nonce = nonce
                block.coherence = coherence
                return block
```

### Difficulty Adjustment

Target 60-second blocks:
```python
def adjust_difficulty(chain, target_time=60):
    recent_blocks = chain.blocks[-10:]
    actual_time = (recent_blocks[-1].timestamp -
                  recent_blocks[0].timestamp) / len(recent_blocks)

    ratio = target_time / actual_time
    new_difficulty = chain.difficulty * ratio

    # Limit adjustment factor
    new_difficulty = max(chain.difficulty * 0.5,
                        min(chain.difficulty * 2.0, new_difficulty))

    return int(new_difficulty)
```

## Storage

### Database Schema

```sql
-- Blocks table
CREATE TABLE blocks (
    height INTEGER PRIMARY KEY,
    hash TEXT UNIQUE,
    prev_hash TEXT,
    timestamp REAL,
    nonce TEXT,
    coherence REAL,
    merkle_root TEXT,
    difficulty INTEGER
);

-- Transactions table
CREATE TABLE transactions (
    txid TEXT PRIMARY KEY,
    block_height INTEGER,
    raw_tx BLOB,
    FOREIGN KEY (block_height) REFERENCES blocks(height)
);

-- UTXO table
CREATE TABLE utxos (
    txid TEXT,
    output_index INTEGER,
    address TEXT,
    amount REAL,
    spent BOOLEAN DEFAULT FALSE,
    PRIMARY KEY (txid, output_index)
);
```

### State Management

```python
class ChainState:
    def __init__(self):
        self.blocks = []           # Full block history
        self.utxo_set = {}        # Current UTXOs
        self.mempool = []         # Pending transactions
        self.headers = []         # Block headers only

    def apply_block(self, block):
        # Update UTXO set
        for tx in block.transactions:
            # Remove spent UTXOs
            for input in tx.inputs:
                key = f"{input.txid}:{input.index}"
                del self.utxo_set[key]

            # Add new UTXOs
            for i, output in enumerate(tx.outputs):
                key = f"{tx.id()}:{i}"
                self.utxo_set[key] = UTXO(
                    txid=tx.id(),
                    index=i,
                    address=output.address,
                    amount=output.amount
                )
```

## Security Considerations

### 1. Double Spend Prevention

- UTXO model inherently prevents double spending
- Mempool checks for conflicting transactions
- Network consensus on transaction ordering

### 2. 51% Attack Resistance

- Coherence is emergent, not parallelizable
- Phase gossip protocol verifies coherence
- Cumulative coherence score for chain selection

### 3. Transaction Malleability

- Transaction IDs exclude signatures
- Witness segregation for signature data
- Strict DER encoding for signatures

## Performance Optimization

### 1. UTXO Index

```python
# Indexed by address for fast balance queries
address_index = {
    "address": [UTXO, UTXO, ...],
    ...
}

# Bloom filters for existence checks
utxo_bloom = BloomFilter(capacity=1000000, error_rate=0.001)
```

### 2. Parallel Validation

```python
def validate_block_parallel(block):
    # Validate transactions in parallel
    with ThreadPoolExecutor() as executor:
        futures = []
        for tx in block.transactions:
            futures.append(executor.submit(validate_transaction, tx))

        results = [f.result() for f in futures]
        return all(results)
```

### 3. Chain Pruning

```python
def prune_old_blocks(chain, keep_recent=1000):
    # Keep only recent blocks and UTXO set
    if len(chain.blocks) > keep_recent:
        # Archive old blocks
        archive_blocks = chain.blocks[:-keep_recent]
        save_to_archive(archive_blocks)

        # Keep only recent blocks in memory
        chain.blocks = chain.blocks[-keep_recent:]

        # UTXO set maintains full state
        return True
```

## Testing

### Unit Tests

```python
# Test block creation
def test_block_creation():
    block = Block(index=1, transactions=[], prev_hash="0"*64)
    assert block.index == 1
    assert block.merkle_root == hash([])

# Test UTXO spending
def test_utxo_spending():
    utxo = UTXO(txid="abc", index=0, address="BC1...", amount=100)
    tx = Transaction()
    tx.add_input(utxo.txid, utxo.index)
    tx.add_output("BC1xyz...", 99)  # 1 BLC fee
    assert validate_transaction(tx)
```

### Integration Tests

```python
# Test full blockchain
def test_blockchain_integration():
    chain = Blockchain()

    # Mine 10 blocks
    for i in range(10):
        block = create_block_with_txs()
        mined = mine_block(block, difficulty=2)
        chain.add_block(mined)

    assert len(chain.blocks) == 11  # Including genesis
    assert chain.is_valid()
```

## API Reference

### Block Methods

```python
block.hash() -> str                    # Calculate block hash
block.is_valid() -> bool              # Validate block
block.serialize() -> bytes            # Serialize for network
Block.deserialize(data) -> Block      # Deserialize from network
```

### Transaction Methods

```python
tx.id() -> str                        # Transaction ID (hash)
tx.sign(private_key) -> None         # Sign transaction
tx.verify() -> bool                  # Verify signature
tx.fee() -> float                    # Calculate transaction fee
```

### Blockchain Methods

```python
chain.add_block(block) -> bool       # Add validated block
chain.get_block(height) -> Block     # Get block by height
chain.get_balance(address) -> float  # Calculate address balance
chain.is_valid() -> bool            # Validate entire chain
chain.get_utxos(address) -> List    # Get UTXOs for address
```

## Future Enhancements

### 1. Smart Contracts
- Stack-based scripting language
- Coherence-based gas model
- State channels for scalability

### 2. Privacy Features
- Confidential transactions
- Ring signatures
- Zero-knowledge proofs

### 3. Scalability
- Sharding based on oscillator groups
- Lightning Network compatibility
- Sidechains with coherence checkpoints

---

*The blockchain module provides the foundation for BloomCoin's distributed consensus, combining UTXO efficiency with Proof-of-Coherence innovation.*