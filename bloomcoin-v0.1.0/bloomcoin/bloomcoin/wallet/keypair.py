"""
BloomCoin Key Pair Management

Ed25519 key generation and management with BIP39 mnemonic support.
"""

from dataclasses import dataclass
from typing import Optional
import os
import hashlib
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey, Ed25519PublicKey
)
from cryptography.hazmat.primitives import serialization

# BIP39 wordlist (first 256 words for demo, full 2048 in production)
BIP39_WORDLIST = [
    "abandon", "ability", "able", "about", "above", "absent", "absorb", "abstract",
    "absurd", "abuse", "access", "accident", "account", "accuse", "achieve", "acid",
    "acoustic", "acquire", "across", "act", "action", "actor", "actress", "actual",
    "adapt", "add", "addict", "address", "adjust", "admit", "adult", "advance",
    "advice", "aerobic", "affair", "afford", "afraid", "again", "age", "agent",
    "agree", "ahead", "aim", "air", "airport", "aisle", "alarm", "album",
    "alcohol", "alert", "alien", "all", "alley", "allow", "almost", "alone",
    "alpha", "already", "also", "alter", "always", "amateur", "amazing", "among",
    "amount", "amused", "analyst", "anchor", "ancient", "anger", "angle", "angry",
    "animal", "ankle", "announce", "annual", "another", "answer", "antenna", "antique",
    "anxiety", "any", "apart", "apology", "appear", "apple", "approve", "april",
    "arch", "arctic", "area", "arena", "argue", "arm", "armed", "armor",
    "army", "around", "arrange", "arrest", "arrive", "arrow", "art", "artefact",
    "artist", "artwork", "ask", "aspect", "assault", "asset", "assist", "assume",
    "asthma", "athlete", "atom", "attack", "attend", "attitude", "attract", "auction",
    "audit", "august", "aunt", "author", "auto", "autumn", "average", "avocado",
    "avoid", "awake", "aware", "away", "awesome", "awful", "awkward", "axis",
    "baby", "bachelor", "bacon", "badge", "bag", "balance", "balcony", "ball",
    "bamboo", "banana", "banner", "bar", "barely", "bargain", "barrel", "base",
    "basic", "basket", "battle", "beach", "bean", "beauty", "because", "become",
    "beef", "before", "begin", "behave", "behind", "believe", "below", "belt",
    "bench", "benefit", "best", "betray", "better", "between", "beyond", "bicycle",
    "bid", "bike", "bind", "biology", "bird", "birth", "bitter", "black",
    "blade", "blame", "blanket", "blast", "bleak", "bless", "blind", "blood",
    "blossom", "blouse", "blue", "blur", "blush", "board", "boat", "body",
    "boil", "bomb", "bone", "bonus", "book", "boost", "border", "boring",
    "borrow", "boss", "bottom", "bounce", "box", "boy", "bracket", "brain",
    "brand", "brass", "brave", "bread", "breeze", "brick", "bridge", "brief",
    "bright", "bring", "brisk", "broccoli", "broken", "bronze", "broom", "brother",
    "brown", "brush", "bubble", "buddy", "budget", "buffalo", "build", "bulb",
    "bulk", "bullet", "bundle", "bunker", "burden", "burger", "burst", "bus",
    "business", "busy", "butter", "buyer", "buzz", "cabbage", "cabin", "cable"
]

# Load full BIP39 wordlist from resources
def load_full_wordlist() -> list:
    """Load the complete BIP39 wordlist."""
    # In production, load from file
    # For now, extend with pattern
    wordlist = BIP39_WORDLIST.copy()

    # Generate remaining words for demonstration
    # In production, use actual BIP39 wordlist file
    while len(wordlist) < 2048:
        wordlist.append(f"word{len(wordlist)}")

    return wordlist


# Use full wordlist
BIP39_WORDLIST_FULL = load_full_wordlist()


@dataclass
class KeyPair:
    """
    Ed25519 key pair for BloomCoin wallet.

    Attributes:
        private_key: 32-byte private key
        public_key: 32-byte public key
        _ed25519_private: Cryptography library key object
    """
    private_key: bytes
    public_key: bytes
    _ed25519_private: Optional[Ed25519PrivateKey] = None

    @classmethod
    def generate(cls, seed: Optional[bytes] = None) -> 'KeyPair':
        """
        Generate new key pair.

        Args:
            seed: Optional 32-byte seed (for deterministic generation)

        Returns:
            New KeyPair
        """
        if seed:
            # Deterministic from seed
            if len(seed) < 32:
                seed = hashlib.sha256(seed).digest()
            private = Ed25519PrivateKey.from_private_bytes(seed[:32])
        else:
            # Random generation
            private = Ed25519PrivateKey.generate()

        public = private.public_key()

        private_bytes = private.private_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PrivateFormat.Raw,
            encryption_algorithm=serialization.NoEncryption()
        )

        public_bytes = public.public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw
        )

        return cls(
            private_key=private_bytes,
            public_key=public_bytes,
            _ed25519_private=private
        )

    @classmethod
    def from_private_key(cls, private_key: bytes) -> 'KeyPair':
        """Reconstruct key pair from private key."""
        if len(private_key) != 32:
            raise ValueError("Private key must be 32 bytes")

        private = Ed25519PrivateKey.from_private_bytes(private_key)
        public = private.public_key()

        public_bytes = public.public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw
        )

        return cls(
            private_key=private_key,
            public_key=public_bytes,
            _ed25519_private=private
        )

    def sign(self, message: bytes) -> bytes:
        """
        Sign a message.

        Returns:
            64-byte Ed25519 signature
        """
        if not self._ed25519_private:
            self._ed25519_private = Ed25519PrivateKey.from_private_bytes(self.private_key)
        return self._ed25519_private.sign(message)

    @staticmethod
    def verify(public_key: bytes, message: bytes, signature: bytes) -> bool:
        """
        Verify a signature.

        Returns:
            True if signature is valid
        """
        try:
            public = Ed25519PublicKey.from_public_bytes(public_key)
            public.verify(signature, message)
            return True
        except Exception:
            return False

    def export_private(self) -> str:
        """Export private key as hex string."""
        return self.private_key.hex()

    def export_public(self) -> str:
        """Export public key as hex string."""
        return self.public_key.hex()


def generate_mnemonic(strength: int = 128) -> str:
    """
    Generate BIP39-compatible mnemonic phrase.

    Args:
        strength: Entropy bits (128 = 12 words, 256 = 24 words)

    Returns:
        Space-separated mnemonic phrase
    """
    if strength not in (128, 160, 192, 224, 256):
        raise ValueError("Invalid strength, must be 128, 160, 192, 224, or 256")

    entropy = os.urandom(strength // 8)

    # Add checksum
    h = hashlib.sha256(entropy).digest()
    checksum_bits = strength // 32

    # Convert to binary string
    entropy_bits = bin(int.from_bytes(entropy, 'big'))[2:].zfill(strength)
    checksum = bin(h[0])[2:].zfill(8)[:checksum_bits]
    all_bits = entropy_bits + checksum

    # Convert to words
    words = []
    for i in range(0, len(all_bits), 11):
        index = int(all_bits[i:i+11], 2)
        words.append(BIP39_WORDLIST_FULL[index])

    return ' '.join(words)


def validate_mnemonic(mnemonic: str) -> bool:
    """
    Validate a mnemonic phrase.

    Args:
        mnemonic: Space-separated mnemonic phrase

    Returns:
        True if valid
    """
    words = mnemonic.split()

    # Check word count
    if len(words) not in (12, 15, 18, 21, 24):
        return False

    # Check all words are in wordlist
    for word in words:
        if word not in BIP39_WORDLIST_FULL:
            return False

    # Convert words to bits
    bits = ''
    for word in words:
        index = BIP39_WORDLIST_FULL.index(word)
        bits += bin(index)[2:].zfill(11)

    # Split entropy and checksum
    entropy_bits = len(words) * 11 * 32 // 33
    checksum_bits = len(bits) - entropy_bits

    entropy = bits[:entropy_bits]
    checksum = bits[entropy_bits:]

    # Verify checksum
    entropy_bytes = int(entropy, 2).to_bytes(entropy_bits // 8, 'big')
    h = hashlib.sha256(entropy_bytes).digest()
    expected_checksum = bin(h[0])[2:].zfill(8)[:checksum_bits]

    return checksum == expected_checksum


def mnemonic_to_seed(mnemonic: str, passphrase: str = '') -> bytes:
    """
    Convert mnemonic to seed.

    Uses PBKDF2-HMAC-SHA512 per BIP39.

    Args:
        mnemonic: Space-separated mnemonic phrase
        passphrase: Optional passphrase

    Returns:
        64-byte seed
    """
    import hashlib

    salt = ('mnemonic' + passphrase).encode('utf-8')
    mnemonic_bytes = mnemonic.encode('utf-8')

    return hashlib.pbkdf2_hmac(
        'sha512',
        mnemonic_bytes,
        salt,
        iterations=2048,
        dklen=64
    )


def keypair_from_mnemonic(mnemonic: str, passphrase: str = '') -> KeyPair:
    """
    Generate key pair from mnemonic.

    Args:
        mnemonic: BIP39 mnemonic phrase
        passphrase: Optional passphrase

    Returns:
        KeyPair derived from mnemonic
    """
    if not validate_mnemonic(mnemonic):
        raise ValueError("Invalid mnemonic phrase")

    seed = mnemonic_to_seed(mnemonic, passphrase)
    return KeyPair.generate(seed=seed[:32])


def generate_hd_path(seed: bytes, path: str = "m/44'/501'/0'/0/0") -> KeyPair:
    """
    Generate key pair from HD path (simplified).

    Note: Full HD wallet support would require BIP32 implementation.
    This is a simplified version for demonstration.

    Args:
        seed: 64-byte seed
        path: HD derivation path

    Returns:
        KeyPair at specified path
    """
    # For now, just use seed directly
    # Full implementation would follow BIP32
    path_hash = hashlib.sha256(path.encode()).digest()
    derived_seed = hashlib.sha256(seed[:32] + path_hash).digest()
    return KeyPair.generate(seed=derived_seed)


def secure_zero(data: bytes):
    """
    Overwrite bytes in memory (security feature).

    Note: Python doesn't guarantee this will work due to
    immutable strings and garbage collection.
    """
    import ctypes
    import sys

    if sys.version_info >= (3, 0):
        # Attempt to overwrite for Python 3
        try:
            ptr = ctypes.cast(id(data), ctypes.POINTER(ctypes.c_char))
            for i in range(len(data)):
                ptr[i] = 0
        except Exception:
            pass  # Best effort