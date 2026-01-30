# BloomCoin Wallet Module - Ed25519 with BIP39

## Overview

The wallet module provides secure key management, address generation, and transaction signing for BloomCoin. It uses **Ed25519** digital signatures for security and **BIP39** mnemonics for user-friendly key backup and recovery.

## Architecture

```
wallet/
├── keypair.py          # Ed25519 key generation
├── address.py         # Address derivation and encoding
├── signer.py          # Transaction signing
├── wallet.py          # Complete wallet management
├── mnemonic.py        # BIP39 mnemonic phrases
├── storage.py         # Secure key storage
└── multi_sig.py       # Multi-signature support
```

## Cryptographic Foundation

### Ed25519 Signatures

BloomCoin uses Ed25519 for digital signatures, providing:
- **128-bit security** level
- **Fast signature** generation and verification
- **Small key sizes** (32 bytes)
- **Deterministic signatures** (no random nonce needed)
- **Resistance** to side-channel attacks

```python
# Key generation
private_key = generate_private_key()  # 32 random bytes
public_key = derive_public_key(private_key)  # Ed25519 point

# Signing
signature = sign_message(message, private_key)  # 64 bytes

# Verification
valid = verify_signature(message, signature, public_key)
```

### Blake2b Hashing

Address derivation uses Blake2b-256:
- **Faster** than SHA-256
- **More secure** than SHA-256
- **No length extension** attacks
- **Built-in keying** support

### Base58Check Encoding

Human-readable addresses use Base58Check:
- **No ambiguous characters** (0, O, I, l)
- **Built-in checksum** for error detection
- **Compact representation** of binary data

## Core Components

### 1. Key Pair Generation (`keypair.py`)

```python
class KeyPair:
    """Ed25519 key pair management."""

    def __init__(self, private_key=None):
        if private_key is None:
            # Generate new random key
            self.private_key = self.generate_private_key()
        else:
            self.private_key = private_key

        # Derive public key
        self.public_key = self.derive_public_key()

    @staticmethod
    def generate_private_key():
        """Generate cryptographically secure private key."""
        # Use OS random source
        return os.urandom(32)

    def derive_public_key(self):
        """Derive Ed25519 public key from private key."""
        # Ed25519 curve operations
        sk = ed25519.SigningKey(self.private_key)
        vk = sk.get_verifying_key()
        return vk.to_bytes()

    def sign(self, message):
        """Sign message with private key."""
        sk = ed25519.SigningKey(self.private_key)
        signature = sk.sign(message)
        return signature

    def verify(self, message, signature):
        """Verify signature with public key."""
        vk = ed25519.VerifyingKey(self.public_key)
        try:
            vk.verify(signature, message)
            return True
        except ed25519.BadSignatureError:
            return False

    @classmethod
    def from_mnemonic(cls, mnemonic, passphrase=""):
        """Generate key pair from BIP39 mnemonic."""
        # Derive seed from mnemonic
        seed = mnemonic_to_seed(mnemonic, passphrase)

        # Use first 32 bytes as private key
        private_key = seed[:32]

        return cls(private_key)

    def to_wif(self):
        """Export private key in Wallet Import Format."""
        # Version byte + private key + checksum
        extended = b'\x80' + self.private_key
        checksum = hashlib.sha256(
            hashlib.sha256(extended).digest()
        ).digest()[:4]

        return base58.b58encode(extended + checksum).decode()

    @classmethod
    def from_wif(cls, wif):
        """Import private key from WIF."""
        decoded = base58.b58decode(wif)

        # Verify checksum
        key_with_version = decoded[:-4]
        checksum = decoded[-4:]
        calculated = hashlib.sha256(
            hashlib.sha256(key_with_version).digest()
        ).digest()[:4]

        if checksum != calculated:
            raise ValueError("Invalid WIF checksum")

        # Extract private key (skip version byte)
        private_key = key_with_version[1:]

        return cls(private_key)
```

### 2. Address Generation (`address.py`)

```python
class Address:
    """BloomCoin address generation and validation."""

    # Address prefixes
    MAINNET_PREFIX = b'BC'  # BloomCoin
    TESTNET_PREFIX = b'TB'  # TestBloom

    @staticmethod
    def from_public_key(public_key, network='mainnet'):
        """Generate address from public key."""
        # Hash public key with Blake2b-256
        blake = hashlib.blake2b(public_key, digest_size=32)
        key_hash = blake.digest()

        # Add network prefix
        prefix = Address.MAINNET_PREFIX if network == 'mainnet' \
                else Address.TESTNET_PREFIX

        # Create address with Base58Check
        address_bytes = prefix + key_hash

        # Add checksum
        checksum = hashlib.sha256(
            hashlib.sha256(address_bytes).digest()
        ).digest()[:4]

        # Encode
        full_address = address_bytes + checksum
        encoded = base58.b58encode(full_address)

        return encoded.decode('utf-8')

    @staticmethod
    def validate(address):
        """Validate address format and checksum."""
        try:
            # Decode from Base58
            decoded = base58.b58decode(address)

            # Verify minimum length
            if len(decoded) < 6:
                return False

            # Split components
            address_bytes = decoded[:-4]
            checksum = decoded[-4:]

            # Verify checksum
            calculated = hashlib.sha256(
                hashlib.sha256(address_bytes).digest()
            ).digest()[:4]

            return checksum == calculated

        except:
            return False

    @staticmethod
    def get_network(address):
        """Determine network from address prefix."""
        decoded = base58.b58decode(address)
        prefix = decoded[:2]

        if prefix == Address.MAINNET_PREFIX:
            return 'mainnet'
        elif prefix == Address.TESTNET_PREFIX:
            return 'testnet'
        else:
            return 'unknown'

class AddressBook:
    """Manage saved addresses."""

    def __init__(self, storage_path='addresses.db'):
        self.storage_path = storage_path
        self.addresses = self.load_addresses()

    def add_address(self, label, address, notes=""):
        """Add address to book."""
        if not Address.validate(address):
            raise ValueError("Invalid address")

        self.addresses[label] = {
            'address': address,
            'notes': notes,
            'created': time.time(),
            'network': Address.get_network(address)
        }

        self.save_addresses()

    def get_address(self, label):
        """Get address by label."""
        return self.addresses.get(label)

    def search_addresses(self, query):
        """Search addresses by label or notes."""
        results = []

        for label, info in self.addresses.items():
            if query.lower() in label.lower() or \
               query.lower() in info['notes'].lower():
                results.append((label, info))

        return results
```

### 3. Transaction Signing (`signer.py`)

```python
class TransactionSigner:
    """Sign and verify transactions."""

    @staticmethod
    def sign_transaction(transaction, private_key):
        """Sign transaction with private key."""
        # Create signing message
        message = TransactionSigner.create_signing_message(transaction)

        # Sign with Ed25519
        keypair = KeyPair(private_key)
        signature = keypair.sign(message)

        # Add signature to transaction
        transaction.signature = signature

        return transaction

    @staticmethod
    def create_signing_message(transaction):
        """Create message to sign (excludes signatures)."""
        # Include all fields except signatures
        data = {
            'inputs': [
                {
                    'txid': inp.txid,
                    'index': inp.index,
                    'amount': inp.amount
                }
                for inp in transaction.inputs
            ],
            'outputs': [
                {
                    'address': out.address,
                    'amount': out.amount
                }
                for out in transaction.outputs
            ],
            'timestamp': transaction.timestamp,
            'memo': transaction.memo
        }

        # Deterministic serialization
        message = json.dumps(data, sort_keys=True)
        return message.encode('utf-8')

    @staticmethod
    def verify_transaction(transaction):
        """Verify all signatures in transaction."""
        # Get signing message
        message = TransactionSigner.create_signing_message(transaction)

        # Verify each input signature
        for i, inp in enumerate(transaction.inputs):
            # Get public key from previous output
            prev_output = get_output(inp.txid, inp.index)
            public_key = get_public_key(prev_output.address)

            # Verify signature
            keypair = KeyPair()
            keypair.public_key = public_key

            if not keypair.verify(message, inp.signature):
                return False, f"Invalid signature on input {i}"

        return True, "All signatures valid"

class TransactionBuilder:
    """Build and prepare transactions."""

    def __init__(self, wallet):
        self.wallet = wallet
        self.inputs = []
        self.outputs = []
        self.fee = 0

    def add_input(self, utxo):
        """Add input (UTXO to spend)."""
        self.inputs.append({
            'txid': utxo.txid,
            'index': utxo.index,
            'amount': utxo.amount,
            'address': utxo.address
        })
        return self

    def add_output(self, address, amount):
        """Add output (recipient)."""
        if not Address.validate(address):
            raise ValueError("Invalid address")

        self.outputs.append({
            'address': address,
            'amount': amount
        })
        return self

    def set_fee(self, fee):
        """Set transaction fee."""
        self.fee = fee
        return self

    def select_utxos(self, amount):
        """Select UTXOs to cover amount + fee."""
        available = self.wallet.get_utxos()

        # Sort by amount (largest first)
        available.sort(key=lambda x: x.amount, reverse=True)

        selected = []
        total = 0

        for utxo in available:
            if total >= amount + self.fee:
                break

            selected.append(utxo)
            total += utxo.amount

        if total < amount + self.fee:
            raise ValueError("Insufficient funds")

        # Add change output if needed
        change = total - amount - self.fee
        if change > 0:
            change_address = self.wallet.get_change_address()
            self.add_output(change_address, change)

        return selected

    def build(self):
        """Build complete transaction."""
        # Create transaction
        tx = Transaction()

        # Add inputs
        for inp in self.inputs:
            tx.add_input(inp['txid'], inp['index'])

        # Add outputs
        for out in self.outputs:
            tx.add_output(out['address'], out['amount'])

        # Sign transaction
        signed_tx = TransactionSigner.sign_transaction(
            tx,
            self.wallet.private_key
        )

        return signed_tx
```

### 4. Complete Wallet (`wallet.py`)

```python
class Wallet:
    """Complete wallet functionality."""

    def __init__(self, private_key=None, network='mainnet'):
        # Initialize key pair
        if private_key:
            self.keypair = KeyPair(private_key)
        else:
            self.keypair = KeyPair()

        self.network = network

        # Generate address
        self.address = Address.from_public_key(
            self.keypair.public_key,
            network
        )

        # Transaction history
        self.transactions = []
        self.utxos = []

        # Address book
        self.address_book = AddressBook()

    @classmethod
    def from_mnemonic(cls, mnemonic, passphrase="", network='mainnet'):
        """Create wallet from BIP39 mnemonic."""
        keypair = KeyPair.from_mnemonic(mnemonic, passphrase)
        wallet = cls(keypair.private_key, network)
        wallet.mnemonic = mnemonic
        return wallet

    @classmethod
    def generate(cls, network='mainnet'):
        """Generate new wallet with mnemonic."""
        # Generate 24-word mnemonic
        mnemonic = generate_mnemonic(strength=256)
        return cls.from_mnemonic(mnemonic, network=network)

    def get_balance(self):
        """Calculate total balance from UTXOs."""
        return sum(utxo.amount for utxo in self.utxos if not utxo.spent)

    def update_utxos(self, blockchain):
        """Update UTXO list from blockchain."""
        self.utxos = blockchain.get_utxos_for_address(self.address)

    def create_transaction(self, recipient, amount, fee=0.001):
        """Create and sign transaction."""
        builder = TransactionBuilder(self)

        # Select UTXOs
        utxos = builder.select_utxos(amount + fee)

        # Add inputs
        for utxo in utxos:
            builder.add_input(utxo)

        # Add recipient output
        builder.add_output(recipient, amount)

        # Set fee
        builder.set_fee(fee)

        # Build and sign
        transaction = builder.build()

        return transaction

    def export_private_key(self):
        """Export private key in WIF format."""
        return self.keypair.to_wif()

    def export_mnemonic(self):
        """Export mnemonic phrase."""
        if hasattr(self, 'mnemonic'):
            return self.mnemonic
        else:
            raise ValueError("Wallet not created from mnemonic")

    def save_to_file(self, filepath, password):
        """Save encrypted wallet to file."""
        # Prepare wallet data
        data = {
            'version': 1,
            'network': self.network,
            'private_key': self.keypair.private_key.hex(),
            'address': self.address,
            'created': time.time()
        }

        # Encrypt with password
        encrypted = encrypt_data(json.dumps(data), password)

        # Save to file
        with open(filepath, 'wb') as f:
            f.write(encrypted)

    @classmethod
    def load_from_file(cls, filepath, password):
        """Load wallet from encrypted file."""
        # Read encrypted data
        with open(filepath, 'rb') as f:
            encrypted = f.read()

        # Decrypt
        decrypted = decrypt_data(encrypted, password)
        data = json.loads(decrypted)

        # Restore wallet
        private_key = bytes.fromhex(data['private_key'])
        wallet = cls(private_key, data['network'])

        return wallet
```

### 5. BIP39 Mnemonic Support (`mnemonic.py`)

```python
class Mnemonic:
    """BIP39 mnemonic phrase generation and validation."""

    def __init__(self, language='english'):
        self.wordlist = load_wordlist(language)
        self.wordlist_set = set(self.wordlist)

    def generate(self, strength=256):
        """Generate mnemonic phrase.

        Args:
            strength: 128 (12 words), 192 (18 words), or 256 (24 words)
        """
        if strength not in [128, 192, 256]:
            raise ValueError("Strength must be 128, 192, or 256")

        # Generate entropy
        entropy = os.urandom(strength // 8)

        # Add checksum
        checksum_length = strength // 32
        h = hashlib.sha256(entropy).digest()
        checksum = bin(h[0])[2:].zfill(8)[:checksum_length]

        # Convert to binary string
        entropy_bits = bin(int.from_bytes(entropy, 'big'))[2:].zfill(strength)
        bits = entropy_bits + checksum

        # Split into 11-bit chunks
        words = []
        for i in range(0, len(bits), 11):
            index = int(bits[i:i+11], 2)
            words.append(self.wordlist[index])

        return ' '.join(words)

    def validate(self, mnemonic):
        """Validate mnemonic phrase."""
        words = mnemonic.split()

        # Check word count
        if len(words) not in [12, 18, 24]:
            return False

        # Check all words in wordlist
        for word in words:
            if word not in self.wordlist_set:
                return False

        # Verify checksum
        # Convert words back to bits
        bits = ''
        for word in words:
            index = self.wordlist.index(word)
            bits += bin(index)[2:].zfill(11)

        # Split entropy and checksum
        checksum_length = len(words) // 3
        entropy_bits = bits[:-checksum_length]
        checksum_bits = bits[-checksum_length:]

        # Calculate expected checksum
        entropy_bytes = int(entropy_bits, 2).to_bytes(len(entropy_bits) // 8, 'big')
        h = hashlib.sha256(entropy_bytes).digest()
        expected_checksum = bin(h[0])[2:].zfill(8)[:checksum_length]

        return checksum_bits == expected_checksum

    def to_seed(self, mnemonic, passphrase=""):
        """Convert mnemonic to seed using PBKDF2."""
        if not self.validate(mnemonic):
            raise ValueError("Invalid mnemonic")

        # PBKDF2 with 2048 iterations
        salt = ("mnemonic" + passphrase).encode('utf-8')
        seed = hashlib.pbkdf2_hmac(
            'sha512',
            mnemonic.encode('utf-8'),
            salt,
            2048
        )

        return seed  # 64 bytes
```

## Security Features

### 1. Secure Key Storage

```python
def encrypt_private_key(private_key, password):
    """Encrypt private key with password."""
    # Derive encryption key from password
    salt = os.urandom(32)
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100000,
    )
    key = kdf.derive(password.encode())

    # Encrypt with AES-256-GCM
    cipher = Cipher(
        algorithms.AES(key),
        modes.GCM(os.urandom(16))
    )
    encryptor = cipher.encryptor()
    ciphertext = encryptor.update(private_key) + encryptor.finalize()

    return salt + encryptor.tag + ciphertext
```

### 2. Multi-Signature Support

```python
class MultiSigWallet:
    """M-of-N multi-signature wallet."""

    def __init__(self, m, n, public_keys):
        self.m = m  # Required signatures
        self.n = n  # Total signers
        self.public_keys = public_keys

        # Generate multi-sig address
        self.address = self.generate_multisig_address()

    def generate_multisig_address(self):
        """Generate P2SH multi-sig address."""
        # Create redeem script
        script = create_multisig_script(self.m, self.public_keys)

        # Hash script
        script_hash = hashlib.blake2b(script, digest_size=20).digest()

        # Create address with special prefix
        return Address.from_script_hash(script_hash)

    def sign_transaction(self, tx, keypair):
        """Add signature to multi-sig transaction."""
        # Sign transaction
        signature = keypair.sign(tx.signing_hash())

        # Add to signature list
        if not hasattr(tx, 'signatures'):
            tx.signatures = []

        tx.signatures.append({
            'public_key': keypair.public_key,
            'signature': signature
        })

        # Check if enough signatures
        if len(tx.signatures) >= self.m:
            tx.complete = True

        return tx
```

### 3. Hardware Wallet Support

```python
class HardwareWallet:
    """Interface for hardware wallet integration."""

    def __init__(self, device_type='ledger'):
        self.device_type = device_type
        self.device = self.connect_device()

    def connect_device(self):
        """Connect to hardware wallet."""
        if self.device_type == 'ledger':
            from ledgerblue.comm import getDongle
            return getDongle(debug=False)
        # Add other hardware wallets

    def get_public_key(self, path="m/44'/501'/0'/0/0"):
        """Get public key from hardware wallet."""
        # Send APDU command
        # ... hardware specific implementation

    def sign_transaction(self, transaction, path="m/44'/501'/0'/0/0"):
        """Sign transaction with hardware wallet."""
        # Prepare transaction for device
        # Send to device for signing
        # Return signature
```

## Usage Examples

### Creating a New Wallet

```python
from bloomcoin.wallet import Wallet

# Generate new wallet with mnemonic
wallet = Wallet.generate()

print(f"Address: {wallet.address}")
print(f"Mnemonic: {wallet.export_mnemonic()}")

# Save encrypted
wallet.save_to_file('my_wallet.dat', password='secure_password')
```

### Restoring from Mnemonic

```python
# Restore from 24 words
mnemonic = "word1 word2 ... word24"
wallet = Wallet.from_mnemonic(mnemonic, passphrase="optional")

print(f"Restored address: {wallet.address}")
```

### Sending a Transaction

```python
# Create and send transaction
recipient = "BC1qxy2kgdygjrsqtzq2n0yrf2493p83kkfjhx0wlh"
amount = 10.5  # BLC

tx = wallet.create_transaction(recipient, amount, fee=0.001)

# Broadcast to network
node.broadcast_transaction(tx)
```

### Multi-Signature Setup

```python
# Create 2-of-3 multi-sig wallet
alice_key = KeyPair().public_key
bob_key = KeyPair().public_key
charlie_key = KeyPair().public_key

multisig = MultiSigWallet(
    m=2, n=3,
    public_keys=[alice_key, bob_key, charlie_key]
)

print(f"Multi-sig address: {multisig.address}")
```

## Testing

### Security Tests

```python
def test_key_generation():
    """Test key pair generation."""
    # Generate 1000 keys and check uniqueness
    keys = set()
    for _ in range(1000):
        kp = KeyPair()
        assert kp.private_key not in keys
        keys.add(kp.private_key)

def test_mnemonic_validation():
    """Test BIP39 mnemonic validation."""
    mnemonic = Mnemonic()

    # Valid mnemonic
    valid = mnemonic.generate(256)
    assert mnemonic.validate(valid)

    # Invalid mnemonic (wrong word)
    invalid = valid.replace('word', 'invalid')
    assert not mnemonic.validate(invalid)

def test_address_validation():
    """Test address validation."""
    # Valid address
    valid = "BC1qxy2kgdygjrsqtzq2n0yrf2493p83kkfjhx0wlh"
    assert Address.validate(valid)

    # Invalid checksum
    invalid = valid[:-1] + 'X'
    assert not Address.validate(invalid)
```

## Configuration

```python
# wallet_config.py
WALLET_CONFIG = {
    'default_network': 'mainnet',
    'mnemonic_strength': 256,  # 24 words
    'pbkdf2_iterations': 100000,
    'default_fee': 0.001,  # BLC

    # Security
    'min_password_length': 12,
    'key_encryption': 'AES-256-GCM',

    # Backup
    'backup_enabled': True,
    'backup_path': '~/.bloomcoin/backups/',

    # Hardware wallets
    'hardware_support': ['ledger', 'trezor'],

    # Multi-sig
    'max_multisig_size': 15
}
```

---

*The BloomCoin wallet provides secure key management with Ed25519 signatures, BIP39 mnemonics, and comprehensive transaction handling for the Proof-of-Coherence blockchain.*