"""
BloomCoin Wallet Module

Key management, address derivation, and transaction signing.
"""

from .keypair import (
    KeyPair,
    generate_mnemonic,
    validate_mnemonic,
    mnemonic_to_seed,
    keypair_from_mnemonic,
    generate_hd_path,
    secure_zero
)

from .address import (
    public_key_to_address,
    address_to_bytes,
    validate_address,
    get_address_network,
    generate_vanity_address,
    format_address_for_display,
    AddressBook,
    base58_encode,
    base58_decode
)

from .signer import (
    Transaction,
    TxInput,
    TxOutput,
    UTXO,
    TransactionSigner,
    TransactionBuilder,
    MultiSigSigner
)

from .wallet import (
    Wallet,
    WalletManager,
    WalletTransaction
)

__all__ = [
    # KeyPair
    'KeyPair',
    'generate_mnemonic',
    'validate_mnemonic',
    'mnemonic_to_seed',
    'keypair_from_mnemonic',
    'generate_hd_path',
    'secure_zero',

    # Address
    'public_key_to_address',
    'address_to_bytes',
    'validate_address',
    'get_address_network',
    'generate_vanity_address',
    'format_address_for_display',
    'AddressBook',
    'base58_encode',
    'base58_decode',

    # Signer
    'Transaction',
    'TxInput',
    'TxOutput',
    'UTXO',
    'TransactionSigner',
    'TransactionBuilder',
    'MultiSigSigner',

    # Wallet
    'Wallet',
    'WalletManager',
    'WalletTransaction'
]