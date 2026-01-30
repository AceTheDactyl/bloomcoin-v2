#!/usr/bin/env python3
"""
BloomCoin Wallet Demonstration

Demonstrates complete wallet functionality including:
- Key generation and mnemonic backup
- Address derivation
- Transaction creation and signing
- Multi-signature support
"""

import sys
from pathlib import Path
import time

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from bloomcoin.wallet import (
    KeyPair,
    generate_mnemonic,
    validate_mnemonic,
    keypair_from_mnemonic,
    public_key_to_address,
    validate_address,
    format_address_for_display,
    Transaction,
    TxInput,
    TxOutput,
    UTXO,
    TransactionSigner,
    TransactionBuilder,
    MultiSigSigner,
    Wallet,
    WalletManager
)
from bloomcoin.constants import PHI, Z_C, INITIAL_REWARD


def demo_keypair_generation():
    """Demonstrate key pair generation."""
    print("\n" + "="*60)
    print("KEY PAIR GENERATION")
    print("="*60)

    # Generate random key pair
    print("\n1. Random key pair generation:")
    kp = KeyPair.generate()
    print(f"   Private key: {kp.export_private()[:16]}...")
    print(f"   Public key:  {kp.export_public()[:16]}...")

    # Generate from seed
    print("\n2. Deterministic generation from seed:")
    seed = b"BloomCoin demo seed - not secure!"
    kp_det = KeyPair.generate(seed=seed)
    print(f"   Private key: {kp_det.export_private()[:16]}...")
    print(f"   Public key:  {kp_det.export_public()[:16]}...")

    # Sign and verify
    print("\n3. Message signing:")
    message = b"BloomCoin: Proof of Coherence"
    signature = kp.sign(message)
    print(f"   Message:   '{message.decode()}'")
    print(f"   Signature: {signature.hex()[:32]}...")
    print(f"   Verified:  {KeyPair.verify(kp.public_key, message, signature)}")

    return kp


def demo_mnemonic():
    """Demonstrate mnemonic functionality."""
    print("\n" + "="*60)
    print("MNEMONIC PHRASE (BIP39)")
    print("="*60)

    # Generate 12-word mnemonic
    print("\n1. Generate 12-word mnemonic:")
    mnemonic_12 = generate_mnemonic(strength=128)
    words_12 = mnemonic_12.split()
    print("   ", " ".join(words_12[:6]))
    print("   ", " ".join(words_12[6:]))
    print(f"   Valid: {validate_mnemonic(mnemonic_12)}")

    # Generate 24-word mnemonic
    print("\n2. Generate 24-word mnemonic:")
    mnemonic_24 = generate_mnemonic(strength=256)
    words_24 = mnemonic_24.split()
    for i in range(0, 24, 6):
        print("   ", " ".join(words_24[i:i+6]))
    print(f"   Valid: {validate_mnemonic(mnemonic_24)}")

    # Derive key from mnemonic
    print("\n3. Derive key pair from mnemonic:")
    kp = keypair_from_mnemonic(mnemonic_12)
    print(f"   Public key: {kp.export_public()[:32]}...")

    # Same mnemonic = same keys
    kp2 = keypair_from_mnemonic(mnemonic_12)
    print(f"   Deterministic: {kp.public_key == kp2.public_key}")

    return mnemonic_12


def demo_addresses():
    """Demonstrate address generation."""
    print("\n" + "="*60)
    print("ADDRESS GENERATION")
    print("="*60)

    kp = KeyPair.generate()

    # Mainnet address
    print("\n1. Mainnet address:")
    addr_main = public_key_to_address(kp.public_key, testnet=False)
    print(f"   Full:     {addr_main}")
    print(f"   Display:  {format_address_for_display(addr_main)}")
    print(f"   Valid:    {validate_address(addr_main)}")

    # Testnet address
    print("\n2. Testnet address:")
    addr_test = public_key_to_address(kp.public_key, testnet=True)
    print(f"   Full:     {addr_test}")
    print(f"   Display:  {format_address_for_display(addr_test)}")
    print(f"   Valid:    {validate_address(addr_test)}")

    # Different keys = different addresses
    print("\n3. Address uniqueness:")
    kp2 = KeyPair.generate()
    addr2 = public_key_to_address(kp2.public_key)
    print(f"   Address 1: {format_address_for_display(addr_main)}")
    print(f"   Address 2: {format_address_for_display(addr2)}")
    print(f"   Different: {addr_main != addr2}")

    return addr_main


def demo_transaction_signing():
    """Demonstrate transaction creation and signing."""
    print("\n" + "="*60)
    print("TRANSACTION SIGNING")
    print("="*60)

    # Create sender and recipient
    sender_kp = KeyPair.generate()
    sender_addr = public_key_to_address(sender_kp.public_key)

    recipient_kp = KeyPair.generate()
    recipient_addr = public_key_to_address(recipient_kp.public_key)

    print(f"\nSender:    {format_address_for_display(sender_addr)}")
    print(f"Recipient: {format_address_for_display(recipient_addr)}")

    # Create transaction
    print("\n1. Create transaction:")
    from bloomcoin.wallet.address import address_to_bytes

    tx = Transaction(
        version=1,
        inputs=[
            TxInput(
                prev_tx=b'previous_transaction_hash_here',
                output_index=0,
                signature=b''  # Empty before signing
            )
        ],
        outputs=[
            TxOutput(
                amount=100000000,  # 1 BLOOM
                address=address_to_bytes(recipient_addr)
            ),
            TxOutput(
                amount=50000000,  # 0.5 BLOOM (change)
                address=address_to_bytes(sender_addr)
            )
        ],
        locktime=0
    )

    print(f"   Inputs:  {len(tx.inputs)}")
    print(f"   Outputs: {len(tx.outputs)}")
    print(f"   Amount:  1.0 BLOOM")
    print(f"   Change:  0.5 BLOOM")

    # Sign transaction
    print("\n2. Sign transaction:")
    signer = TransactionSigner(sender_kp)
    signed_tx = signer.sign_transaction(tx)

    print(f"   Signature: {signed_tx.inputs[0].signature.hex()[:32]}...")
    print(f"   TxID:      {signed_tx.txid[:16]}...")

    # Verify signature
    print("\n3. Verify signature:")
    message = tx.serialize_for_signing()
    verified = KeyPair.verify(
        sender_kp.public_key,
        message,
        signed_tx.inputs[0].signature
    )
    print(f"   Valid: {verified}")

    return signed_tx


def demo_transaction_builder():
    """Demonstrate transaction builder."""
    print("\n" + "="*60)
    print("TRANSACTION BUILDER")
    print("="*60)

    # Create wallet key
    wallet_kp = KeyPair.generate()
    wallet_addr = public_key_to_address(wallet_kp.public_key)

    print(f"\nWallet: {format_address_for_display(wallet_addr)}")

    # Create UTXOs (unspent outputs)
    print("\n1. Available UTXOs:")
    from bloomcoin.wallet.address import address_to_bytes

    utxos = [
        UTXO(
            txid=b'tx_hash_1',
            index=0,
            amount=300000000,  # 3 BLOOM
            address=address_to_bytes(wallet_addr)
        ),
        UTXO(
            txid=b'tx_hash_2',
            index=1,
            amount=150000000,  # 1.5 BLOOM
            address=address_to_bytes(wallet_addr)
        ),
        UTXO(
            txid=b'tx_hash_3',
            index=0,
            amount=50000000,  # 0.5 BLOOM
            address=address_to_bytes(wallet_addr)
        )
    ]

    total = sum(u.amount for u in utxos)
    print(f"   Total: {total/100000000:.1f} BLOOM ({len(utxos)} UTXOs)")
    for i, utxo in enumerate(utxos):
        print(f"   {i+1}. {utxo.amount/100000000:.1f} BLOOM")

    # Build transaction
    print("\n2. Build transaction:")
    builder = TransactionBuilder(utxos=utxos, keypair=wallet_kp, fee_rate=1)

    recipient = public_key_to_address(KeyPair.generate().public_key)
    print(f"   To: {format_address_for_display(recipient)}")
    print(f"   Amount: 2.5 BLOOM")

    tx = builder.build_and_sign([(recipient, 250000000)])

    if tx:
        print(f"   ✓ Transaction built")
        print(f"   Inputs:  {len(tx.inputs)}")
        print(f"   Outputs: {len(tx.outputs)}")
        print(f"   TxID:    {tx.txid[:16]}...")

        # Calculate fee
        fee = total - sum(out.amount for out in tx.outputs)
        print(f"   Fee:     {fee} satoshis")


def demo_multisig():
    """Demonstrate multi-signature functionality."""
    print("\n" + "="*60)
    print("MULTI-SIGNATURE (2-of-3)")
    print("="*60)

    # Create 3 key pairs (3 parties)
    print("\n1. Create 3 parties:")
    alice = KeyPair.generate()
    bob = KeyPair.generate()
    carol = KeyPair.generate()

    alice_addr = public_key_to_address(alice.public_key)
    bob_addr = public_key_to_address(bob.public_key)
    carol_addr = public_key_to_address(carol.public_key)

    print(f"   Alice: {format_address_for_display(alice_addr)}")
    print(f"   Bob:   {format_address_for_display(bob_addr)}")
    print(f"   Carol: {format_address_for_display(carol_addr)}")

    # Create 2-of-3 multisig
    print("\n2. Create 2-of-3 multisig:")
    multisig = MultiSigSigner(
        threshold=2,
        public_keys=[alice.public_key, bob.public_key, carol.public_key]
    )

    multisig_addr = multisig.create_multisig_address()
    print(f"   Address: {multisig_addr}")
    print(f"   Threshold: 2 of 3 signatures required")

    # Sign message
    print("\n3. Sign message with 2 parties:")
    message = b"Approve multisig transaction"

    sig_alice = alice.sign(message)
    sig_bob = bob.sign(message)

    print(f"   Alice signed: ✓")
    print(f"   Bob signed:   ✓")
    print(f"   Carol signed: ✗")

    # Verify
    valid = multisig.verify_multisig(message, [sig_alice, sig_bob])
    print(f"   Valid (2 sigs): {valid}")

    # Try with only 1 signature
    valid_1 = multisig.verify_multisig(message, [sig_alice])
    print(f"   Valid (1 sig):  {valid_1}")


def demo_wallet():
    """Demonstrate complete wallet functionality."""
    print("\n" + "="*60)
    print("COMPLETE WALLET")
    print("="*60)

    # Create wallet
    print("\n1. Create new wallet:")
    wallet, mnemonic = Wallet.create("demo_wallet", testnet=True)

    print(f"   Name:    {wallet.name}")
    print(f"   Network: {'testnet' if wallet.testnet else 'mainnet'}")
    print(f"   Address: {wallet.address}")
    print(f"   Balance: {wallet.balance_bloom} BLOOM")

    # Show mnemonic
    print("\n2. Backup mnemonic (12 words):")
    words = mnemonic.split()
    print("   ", " ".join(words[:6]))
    print("   ", " ".join(words[6:]))

    # Simulate receiving funds
    print("\n3. Receive funds:")
    from bloomcoin.wallet.address import address_to_bytes

    # Add some UTXOs
    wallet.utxos = [
        UTXO(
            txid=b'received_tx_1',
            index=0,
            amount=int(INITIAL_REWARD),  # Use initial block reward
            address=address_to_bytes(wallet.address)
        )
    ]

    print(f"   Received: {INITIAL_REWARD/100000000:.8f} BLOOM")
    print(f"   Balance:  {wallet.balance_bloom:.8f} BLOOM")

    # Send transaction (simulated)
    print("\n4. Send transaction:")
    recipient = public_key_to_address(KeyPair.generate().public_key, testnet=True)

    print(f"   To:     {format_address_for_display(recipient)}")
    print(f"   Amount: 1.0 BLOOM")

    # Would normally broadcast to network
    # txid = wallet.send_bloom(recipient, 1.0)

    # Wallet info
    print("\n5. Wallet info:")
    info = wallet.get_info()
    for key, value in info.items():
        if key not in ['public_key']:  # Skip long values
            print(f"   {key:15} {value}")


def demo_wallet_manager():
    """Demonstrate wallet manager."""
    print("\n" + "="*60)
    print("WALLET MANAGER")
    print("="*60)

    # Create temporary directory for demo
    import tempfile
    temp_dir = Path(tempfile.mkdtemp())

    print(f"\nData directory: {temp_dir}")

    # Create manager
    manager = WalletManager(temp_dir)

    # Create multiple wallets
    print("\n1. Create wallets:")
    wallet1, mnemonic1 = manager.create_wallet("alice")
    wallet2, mnemonic2 = manager.create_wallet("bob", testnet=True)

    print(f"   Created 'alice' (mainnet)")
    print(f"   Created 'bob' (testnet)")

    # List wallets
    print("\n2. List wallets:")
    wallets = manager.list_wallets()
    for name in wallets:
        wallet = manager.get_wallet(name)
        if wallet:
            print(f"   {name}: {format_address_for_display(wallet.address)}")

    # Clean up
    import shutil
    shutil.rmtree(temp_dir)


def main():
    """Run all demonstrations."""
    print("\n" + "="*60)
    print("BLOOMCOIN WALLET DEMONSTRATION")
    print("="*60)
    print(f"Golden ratio φ = {PHI:.10f}")
    print(f"Critical coherence z_c = {Z_C:.10f}")
    print(f"Initial reward = {INITIAL_REWARD/100000000:.8f} BLOOM")

    # Run demos
    demo_keypair_generation()
    demo_mnemonic()
    demo_addresses()
    demo_transaction_signing()
    demo_transaction_builder()
    demo_multisig()
    demo_wallet()
    demo_wallet_manager()

    print("\n" + "="*60)
    print("DEMONSTRATION COMPLETE")
    print("="*60)
    print("\nKey Features Demonstrated:")
    print("  ✓ Ed25519 key generation")
    print("  ✓ BIP39 mnemonic phrases")
    print("  ✓ Blake2b address derivation")
    print("  ✓ Transaction signing")
    print("  ✓ UTXO management")
    print("  ✓ Multi-signature support")
    print("  ✓ Complete wallet functionality")
    print("\nYour keys, your coins! 🌸")


if __name__ == '__main__':
    main()