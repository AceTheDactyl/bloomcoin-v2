"""
BloomCoin Address Derivation

Derives addresses from public keys using Blake2b hashing and Base58Check encoding.
"""

import hashlib
from typing import Optional

# Address prefix bytes
MAINNET_PREFIX = b'\x00'  # 'B' addresses
TESTNET_PREFIX = b'\x6f'  # 't' addresses

# Base58 alphabet (excludes 0, O, I, l to avoid confusion)
BASE58_ALPHABET = '123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz'


def base58_encode(data: bytes) -> str:
    """
    Base58 encode bytes.

    Args:
        data: Bytes to encode

    Returns:
        Base58 encoded string
    """
    # Count leading zeros
    leading_zeros = 0
    for byte in data:
        if byte == 0:
            leading_zeros += 1
        else:
            break

    # Convert to integer
    num = int.from_bytes(data, 'big')

    # Convert to base58
    result = ''
    while num > 0:
        num, remainder = divmod(num, 58)
        result = BASE58_ALPHABET[remainder] + result

    # Add leading '1's for zeros
    return '1' * leading_zeros + result


def base58_decode(s: str) -> bytes:
    """
    Base58 decode string to bytes.

    Args:
        s: Base58 encoded string

    Returns:
        Decoded bytes
    """
    # Count leading '1's
    leading_ones = 0
    for char in s:
        if char == '1':
            leading_ones += 1
        else:
            break

    # Convert from base58
    num = 0
    for char in s:
        if char not in BASE58_ALPHABET:
            raise ValueError(f"Invalid Base58 character: {char}")
        num = num * 58 + BASE58_ALPHABET.index(char)

    # Convert to bytes
    if num == 0:
        result = b''
    else:
        result = num.to_bytes((num.bit_length() + 7) // 8, 'big')

    # Add leading zeros
    return b'\x00' * leading_ones + result


def public_key_to_address(
    public_key: bytes,
    testnet: bool = False
) -> str:
    """
    Derive address from public key.

    Format:
        1. Blake2b-256(public_key) -> 32 bytes
        2. Take first 20 bytes
        3. Add version prefix
        4. Base58Check encode

    Args:
        public_key: 32-byte Ed25519 public key
        testnet: Use testnet prefix

    Returns:
        Base58Check encoded address (starts with 'B' or 't')
    """
    if len(public_key) != 32:
        raise ValueError("Public key must be 32 bytes")

    # Hash public key
    h = hashlib.blake2b(public_key, digest_size=32).digest()

    # Take first 20 bytes (160 bits)
    payload = h[:20]

    # Add prefix
    prefix = TESTNET_PREFIX if testnet else MAINNET_PREFIX
    versioned = prefix + payload

    # Checksum (first 4 bytes of double-Blake2b)
    checksum = hashlib.blake2b(
        hashlib.blake2b(versioned, digest_size=32).digest(),
        digest_size=32
    ).digest()[:4]

    # Base58 encode
    return base58_encode(versioned + checksum)


def address_to_bytes(address: str) -> bytes:
    """
    Decode address to raw bytes.

    Args:
        address: Base58Check encoded address

    Returns:
        20-byte address payload (without prefix/checksum)

    Raises:
        ValueError: If address is invalid
    """
    try:
        decoded = base58_decode(address)
    except Exception as e:
        raise ValueError(f"Invalid address encoding: {e}")

    if len(decoded) < 25:  # 1 byte prefix + 20 bytes payload + 4 bytes checksum
        raise ValueError("Address too short")

    # Verify checksum
    payload = decoded[:-4]
    checksum = decoded[-4:]
    expected = hashlib.blake2b(
        hashlib.blake2b(payload, digest_size=32).digest(),
        digest_size=32
    ).digest()[:4]

    if checksum != expected:
        raise ValueError("Invalid address checksum")

    # Return payload without prefix
    return payload[1:]


def validate_address(address: str, testnet: Optional[bool] = None) -> bool:
    """
    Check if address is valid.

    Args:
        address: Address to validate
        testnet: If specified, also check if address matches network

    Returns:
        True if address is valid
    """
    try:
        decoded = base58_decode(address)

        if len(decoded) < 25:
            return False

        # Check prefix if network specified
        if testnet is not None:
            expected_prefix = TESTNET_PREFIX if testnet else MAINNET_PREFIX
            if decoded[0:1] != expected_prefix:
                return False

        # Verify checksum
        payload = decoded[:-4]
        checksum = decoded[-4:]
        expected = hashlib.blake2b(
            hashlib.blake2b(payload, digest_size=32).digest(),
            digest_size=32
        ).digest()[:4]

        return checksum == expected

    except Exception:
        return False


def get_address_network(address: str) -> Optional[str]:
    """
    Determine network type from address.

    Args:
        address: Address to check

    Returns:
        'mainnet', 'testnet', or None if invalid
    """
    try:
        decoded = base58_decode(address)

        if len(decoded) < 25:
            return None

        prefix = decoded[0:1]

        if prefix == MAINNET_PREFIX:
            return 'mainnet'
        elif prefix == TESTNET_PREFIX:
            return 'testnet'
        else:
            return None

    except Exception:
        return None


def generate_vanity_address(
    prefix: str,
    testnet: bool = False,
    max_attempts: int = 1000000
) -> Optional[tuple]:
    """
    Generate a vanity address with desired prefix.

    Warning: This can take a very long time for long prefixes!

    Args:
        prefix: Desired address prefix (after network byte)
        testnet: Generate testnet address
        max_attempts: Maximum attempts before giving up

    Returns:
        (keypair, address) if found, None if not found
    """
    from .keypair import KeyPair

    # Validate prefix contains only valid Base58 characters
    for char in prefix:
        if char not in BASE58_ALPHABET:
            raise ValueError(f"Invalid character in prefix: {char}")

    # Expected start character
    expected_start = 't' if testnet else 'B'

    for _ in range(max_attempts):
        # Generate random keypair
        keypair = KeyPair.generate()

        # Derive address
        address = public_key_to_address(keypair.public_key, testnet)

        # Check if it matches desired prefix
        # Skip the network indicator character
        if address[1:].startswith(prefix):
            return keypair, address

    return None


def batch_validate_addresses(addresses: list, testnet: Optional[bool] = None) -> dict:
    """
    Validate multiple addresses efficiently.

    Args:
        addresses: List of addresses to validate
        testnet: If specified, also check network type

    Returns:
        Dict mapping address to validation result
    """
    results = {}

    for address in addresses:
        results[address] = validate_address(address, testnet)

    return results


def format_address_for_display(address: str, abbreviate: bool = True) -> str:
    """
    Format address for user display.

    Args:
        address: Address to format
        abbreviate: Whether to abbreviate middle portion

    Returns:
        Formatted address string
    """
    if not validate_address(address):
        return "Invalid Address"

    if abbreviate and len(address) > 20:
        # Show first 6 and last 6 characters
        return f"{address[:6]}...{address[-6:]}"

    # Add spacing for readability
    # Group in sets of 4 characters
    formatted = ""
    for i in range(0, len(address), 4):
        if formatted:
            formatted += " "
        formatted += address[i:i+4]

    return formatted


class AddressBook:
    """
    Simple address book for managing known addresses.
    """

    def __init__(self):
        self.entries = {}

    def add(self, label: str, address: str, testnet: bool = False):
        """
        Add address to book.

        Args:
            label: Human-readable label
            address: BloomCoin address
            testnet: Whether this is a testnet address

        Raises:
            ValueError: If address is invalid
        """
        if not validate_address(address, testnet):
            raise ValueError(f"Invalid address: {address}")

        self.entries[label] = {
            'address': address,
            'testnet': testnet,
            'network': get_address_network(address)
        }

    def get(self, label: str) -> Optional[str]:
        """Get address by label."""
        entry = self.entries.get(label)
        return entry['address'] if entry else None

    def list_entries(self, testnet: Optional[bool] = None) -> list:
        """
        List all entries, optionally filtered by network.

        Args:
            testnet: Filter by network type

        Returns:
            List of (label, address) tuples
        """
        results = []

        for label, entry in self.entries.items():
            if testnet is None or entry['testnet'] == testnet:
                results.append((label, entry['address']))

        return results

    def export_json(self) -> str:
        """Export address book as JSON."""
        import json
        return json.dumps(self.entries, indent=2)

    def import_json(self, data: str):
        """Import address book from JSON."""
        import json
        imported = json.loads(data)

        for label, entry in imported.items():
            self.add(label, entry['address'], entry.get('testnet', False))