"""
Test suite for BloomCoin wallet module.
"""

import unittest
import tempfile
import shutil
from pathlib import Path
import os
import json

from .keypair import (
    KeyPair,
    generate_mnemonic,
    validate_mnemonic,
    mnemonic_to_seed,
    keypair_from_mnemonic
)
from .address import (
    public_key_to_address,
    address_to_bytes,
    validate_address,
    get_address_network,
    base58_encode,
    base58_decode,
    AddressBook
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
from .wallet import Wallet, WalletManager, WalletTransaction


class TestKeyPair(unittest.TestCase):
    """Test key pair functionality."""

    def test_generate_keypair(self):
        """Test key pair generation."""
        kp = KeyPair.generate()
        self.assertEqual(len(kp.private_key), 32)
        self.assertEqual(len(kp.public_key), 32)
        self.assertIsNotNone(kp._ed25519_private)

    def test_deterministic_generation(self):
        """Test deterministic key generation from seed."""
        seed = b'test seed for deterministic generation'
        kp1 = KeyPair.generate(seed=seed)
        kp2 = KeyPair.generate(seed=seed)

        self.assertEqual(kp1.private_key, kp2.private_key)
        self.assertEqual(kp1.public_key, kp2.public_key)

    def test_from_private_key(self):
        """Test reconstruction from private key."""
        kp1 = KeyPair.generate()
        kp2 = KeyPair.from_private_key(kp1.private_key)

        self.assertEqual(kp1.private_key, kp2.private_key)
        self.assertEqual(kp1.public_key, kp2.public_key)

    def test_sign_verify(self):
        """Test message signing and verification."""
        kp = KeyPair.generate()
        message = b'Test message for signing'

        signature = kp.sign(message)
        self.assertEqual(len(signature), 64)

        # Verify with correct key
        self.assertTrue(KeyPair.verify(kp.public_key, message, signature))

        # Verify with wrong key
        wrong_kp = KeyPair.generate()
        self.assertFalse(KeyPair.verify(wrong_kp.public_key, message, signature))

        # Verify with wrong message
        self.assertFalse(KeyPair.verify(kp.public_key, b'wrong message', signature))


class TestMnemonic(unittest.TestCase):
    """Test mnemonic functionality."""

    def test_generate_mnemonic_12_words(self):
        """Test 12-word mnemonic generation."""
        mnemonic = generate_mnemonic(strength=128)
        words = mnemonic.split()
        self.assertEqual(len(words), 12)
        self.assertTrue(validate_mnemonic(mnemonic))

    def test_generate_mnemonic_24_words(self):
        """Test 24-word mnemonic generation."""
        mnemonic = generate_mnemonic(strength=256)
        words = mnemonic.split()
        self.assertEqual(len(words), 24)
        self.assertTrue(validate_mnemonic(mnemonic))

    def test_validate_mnemonic(self):
        """Test mnemonic validation."""
        # Valid mnemonic (example, would need actual wordlist)
        valid = generate_mnemonic()
        self.assertTrue(validate_mnemonic(valid))

        # Invalid - wrong word count
        invalid1 = "word1 word2 word3"
        self.assertFalse(validate_mnemonic(invalid1))

        # Invalid - not in wordlist
        invalid2 = "xxxx yyyy zzzz aaaa bbbb cccc dddd eeee ffff gggg hhhh iiii"
        self.assertFalse(validate_mnemonic(invalid2))

    def test_mnemonic_to_seed(self):
        """Test mnemonic to seed conversion."""
        mnemonic = "abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon about"
        seed = mnemonic_to_seed(mnemonic)
        self.assertEqual(len(seed), 64)

        # With passphrase
        seed_with_pass = mnemonic_to_seed(mnemonic, "TREZOR")
        self.assertEqual(len(seed_with_pass), 64)
        self.assertNotEqual(seed, seed_with_pass)

    def test_keypair_from_mnemonic(self):
        """Test key pair generation from mnemonic."""
        mnemonic = generate_mnemonic()

        # Generate twice - should be deterministic
        kp1 = keypair_from_mnemonic(mnemonic)
        kp2 = keypair_from_mnemonic(mnemonic)

        self.assertEqual(kp1.private_key, kp2.private_key)
        self.assertEqual(kp1.public_key, kp2.public_key)


class TestAddress(unittest.TestCase):
    """Test address functionality."""

    def test_address_generation(self):
        """Test address generation from public key."""
        kp = KeyPair.generate()

        # Mainnet address
        addr_main = public_key_to_address(kp.public_key, testnet=False)
        self.assertTrue(addr_main.startswith('1') or addr_main.startswith('B'))
        self.assertTrue(validate_address(addr_main))

        # Testnet address
        addr_test = public_key_to_address(kp.public_key, testnet=True)
        self.assertTrue(validate_address(addr_test))

    def test_address_validation(self):
        """Test address validation."""
        kp = KeyPair.generate()
        addr = public_key_to_address(kp.public_key)

        # Valid address
        self.assertTrue(validate_address(addr))

        # Invalid - modified checksum
        invalid = addr[:-1] + 'X'
        self.assertFalse(validate_address(invalid))

        # Invalid - too short
        self.assertFalse(validate_address('BShort'))

    def test_address_roundtrip(self):
        """Test address encoding and decoding."""
        kp = KeyPair.generate()
        addr = public_key_to_address(kp.public_key)

        # Decode address
        addr_bytes = address_to_bytes(addr)
        self.assertEqual(len(addr_bytes), 20)

        # Should be deterministic
        addr2 = public_key_to_address(kp.public_key)
        self.assertEqual(addr, addr2)

    def test_base58_encoding(self):
        """Test Base58 encoding/decoding."""
        data = b'Hello, BloomCoin!'

        encoded = base58_encode(data)
        decoded = base58_decode(encoded)

        self.assertEqual(data, decoded)

        # Test with leading zeros
        data_zeros = b'\x00\x00' + data
        encoded_zeros = base58_encode(data_zeros)
        self.assertTrue(encoded_zeros.startswith('11'))
        decoded_zeros = base58_decode(encoded_zeros)
        self.assertEqual(data_zeros, decoded_zeros)

    def test_address_book(self):
        """Test address book functionality."""
        book = AddressBook()

        kp1 = KeyPair.generate()
        kp2 = KeyPair.generate()
        addr1 = public_key_to_address(kp1.public_key)
        addr2 = public_key_to_address(kp2.public_key, testnet=True)

        # Add addresses
        book.add("Alice", addr1, testnet=False)
        book.add("Bob", addr2, testnet=True)

        # Get addresses
        self.assertEqual(book.get("Alice"), addr1)
        self.assertEqual(book.get("Bob"), addr2)
        self.assertIsNone(book.get("Charlie"))

        # List entries
        mainnet_entries = book.list_entries(testnet=False)
        self.assertEqual(len(mainnet_entries), 1)
        self.assertEqual(mainnet_entries[0], ("Alice", addr1))

        # Export/import
        exported = book.export_json()
        book2 = AddressBook()
        book2.import_json(exported)
        self.assertEqual(book2.get("Alice"), addr1)


class TestTransactionSigning(unittest.TestCase):
    """Test transaction signing."""

    def test_transaction_creation(self):
        """Test transaction creation."""
        tx = Transaction(
            version=1,
            inputs=[
                TxInput(
                    prev_tx=b'prev_tx_hash_1',
                    output_index=0,
                    signature=b''
                )
            ],
            outputs=[
                TxOutput(
                    amount=50000000,
                    address=b'recipient_address_20'
                )
            ],
            locktime=0
        )

        self.assertEqual(tx.version, 1)
        self.assertEqual(len(tx.inputs), 1)
        self.assertEqual(len(tx.outputs), 1)

        # Test serialization
        serialized = tx.serialize_for_signing()
        self.assertIsInstance(serialized, bytes)
        self.assertGreater(len(serialized), 0)

    def test_transaction_signing(self):
        """Test transaction signing."""
        kp = KeyPair.generate()
        signer = TransactionSigner(kp)

        # Create transaction
        tx = Transaction(
            version=1,
            inputs=[
                TxInput(prev_tx=b'prev1', output_index=0),
                TxInput(prev_tx=b'prev2', output_index=1)
            ],
            outputs=[
                TxOutput(amount=50000000, address=b'addr1'),
                TxOutput(amount=25000000, address=b'addr2')
            ],
            locktime=0
        )

        # Sign transaction
        signed_tx = signer.sign_transaction(tx)

        # Check signatures are present
        for inp in signed_tx.inputs:
            self.assertEqual(len(inp.signature), 64)

        # Verify signatures
        message = tx.serialize_for_signing()
        for inp in signed_tx.inputs:
            self.assertTrue(KeyPair.verify(kp.public_key, message, inp.signature))

    def test_transaction_builder(self):
        """Test transaction builder."""
        kp = KeyPair.generate()
        addr = public_key_to_address(kp.public_key)
        addr_bytes = address_to_bytes(addr)

        # Create UTXOs
        utxos = [
            UTXO(txid=b'tx1', index=0, amount=100000000, address=addr_bytes),
            UTXO(txid=b'tx2', index=1, amount=50000000, address=addr_bytes)
        ]

        builder = TransactionBuilder(utxos=utxos, keypair=kp, fee_rate=1)

        # Build transaction
        recipient_kp = KeyPair.generate()
        recipient_addr = public_key_to_address(recipient_kp.public_key)

        tx = builder.build([(recipient_addr, 75000000)])
        self.assertIsNotNone(tx)

        # Check inputs and outputs
        self.assertGreaterEqual(len(tx.inputs), 1)
        self.assertGreaterEqual(len(tx.outputs), 1)

        # First output should be to recipient
        self.assertEqual(tx.outputs[0].amount, 75000000)

    def test_multisig(self):
        """Test multi-signature functionality."""
        # Create 3 key pairs
        kps = [KeyPair.generate() for _ in range(3)]
        public_keys = [kp.public_key for kp in kps]

        # Create 2-of-3 multisig
        multisig = MultiSigSigner(threshold=2, public_keys=public_keys)

        # Create multisig address
        addr = multisig.create_multisig_address()
        self.assertTrue(len(addr) > 0)

        # Sign message with 2 keys
        message = b'Test multisig message'
        sig1 = kps[0].sign(message)
        sig2 = kps[1].sign(message)

        # Verify with 2 signatures (should pass)
        self.assertTrue(multisig.verify_multisig(message, [sig1, sig2]))

        # Verify with 1 signature (should fail)
        self.assertFalse(multisig.verify_multisig(message, [sig1]))


class TestWallet(unittest.TestCase):
    """Test wallet functionality."""

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)

    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)

    def test_wallet_creation(self):
        """Test wallet creation."""
        wallet, mnemonic = Wallet.create("test_wallet", testnet=True)

        self.assertEqual(wallet.name, "test_wallet")
        self.assertTrue(wallet.testnet)
        self.assertIsNotNone(wallet.address)
        self.assertEqual(wallet.balance, 0)
        self.assertTrue(validate_mnemonic(mnemonic))

    def test_wallet_restoration(self):
        """Test wallet restoration from mnemonic."""
        # Create wallet
        wallet1, mnemonic = Wallet.create("wallet1")

        # Restore from same mnemonic
        wallet2 = Wallet.restore("wallet2", mnemonic)

        # Should have same keys
        self.assertEqual(wallet1.keypair.private_key, wallet2.keypair.private_key)
        self.assertEqual(wallet1.keypair.public_key, wallet2.keypair.public_key)
        self.assertEqual(wallet1.address, wallet2.address)

    def test_wallet_save_load(self):
        """Test wallet save and load."""
        wallet1, _ = Wallet.create("test")

        # Add some data
        wallet1.address_book.add("Alice", wallet1.address)
        wallet1.transaction_history.append(WalletTransaction(
            txid="test_tx",
            timestamp=1234567890,
            amount=100000000,
            fee=1000,
            address="test_addr",
            memo="Test transaction"
        ))

        # Save wallet
        wallet_path = self.temp_path / "test.wallet"
        wallet1.save(wallet_path)

        # Load wallet
        wallet2 = Wallet.load(wallet_path)

        # Check data preserved
        self.assertEqual(wallet1.name, wallet2.name)
        self.assertEqual(wallet1.address, wallet2.address)
        self.assertEqual(wallet1.keypair.private_key, wallet2.keypair.private_key)
        self.assertEqual(wallet2.address_book.get("Alice"), wallet1.address)
        self.assertEqual(len(wallet2.transaction_history), 1)

    def test_wallet_manager(self):
        """Test wallet manager."""
        manager = WalletManager(self.temp_path)

        # Create wallets
        wallet1, mnemonic1 = manager.create_wallet("wallet1")
        wallet2, mnemonic2 = manager.create_wallet("wallet2", testnet=True)

        # List wallets
        wallets = manager.list_wallets()
        self.assertIn("wallet1", wallets)
        self.assertIn("wallet2", wallets)

        # Get wallet
        w1 = manager.get_wallet("wallet1")
        self.assertIsNotNone(w1)
        self.assertEqual(w1.name, "wallet1")

        # Load wallet
        manager2 = WalletManager(self.temp_path)
        w1_loaded = manager2.load_wallet("wallet1")
        self.assertEqual(w1_loaded.address, wallet1.address)

    def test_wallet_utxos(self):
        """Test UTXO management."""
        wallet, _ = Wallet.create("test")
        addr_bytes = address_to_bytes(wallet.address)

        # Add UTXOs
        wallet.utxos = [
            UTXO(txid=b'tx1', index=0, amount=100000000, address=addr_bytes),
            UTXO(txid=b'tx2', index=1, amount=50000000, address=addr_bytes),
            UTXO(txid=b'tx3', index=0, amount=25000000, address=addr_bytes)
        ]

        # Check balance
        self.assertEqual(wallet.balance, 175000000)
        self.assertEqual(wallet.balance_bloom, 1.75)

    def test_wallet_info(self):
        """Test wallet info."""
        wallet, _ = Wallet.create("info_test")
        info = wallet.get_info()

        self.assertEqual(info['name'], "info_test")
        self.assertIn('address', info)
        self.assertIn('public_key', info)
        self.assertEqual(info['network'], 'mainnet')
        self.assertEqual(info['balance'], 0)
        self.assertEqual(info['balance_bloom'], 0.0)


if __name__ == '__main__':
    unittest.main()