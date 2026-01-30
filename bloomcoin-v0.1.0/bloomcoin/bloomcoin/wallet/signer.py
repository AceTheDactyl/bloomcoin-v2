"""
BloomCoin Transaction Signing

Signs transactions with wallet keys using Ed25519 signatures.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Tuple
import hashlib
from .keypair import KeyPair
from .address import public_key_to_address, address_to_bytes


# Transaction structures (simplified versions if blockchain module not available)
@dataclass
class TxInput:
    """Transaction input."""
    prev_tx: bytes  # Previous transaction hash
    output_index: int  # Index in previous transaction
    signature: bytes = b''  # Ed25519 signature


@dataclass
class TxOutput:
    """Transaction output."""
    amount: int  # Amount in smallest units
    address: bytes  # 20-byte recipient address


@dataclass
class Transaction:
    """BloomCoin transaction."""
    version: int
    inputs: List[TxInput]
    outputs: List[TxOutput]
    locktime: int = 0

    def serialize_for_signing(self) -> bytes:
        """
        Serialize transaction for signing.

        Excludes signatures from inputs.

        Returns:
            Bytes to be signed
        """
        data = b''

        # Version (4 bytes)
        data += self.version.to_bytes(4, 'little')

        # Number of inputs (variable length encoding)
        data += len(self.inputs).to_bytes(1, 'big')

        # Inputs (without signatures)
        for inp in self.inputs:
            data += inp.prev_tx
            data += inp.output_index.to_bytes(4, 'little')
            # Skip signature for signing

        # Number of outputs
        data += len(self.outputs).to_bytes(1, 'big')

        # Outputs
        for out in self.outputs:
            data += out.amount.to_bytes(8, 'little')
            data += out.address

        # Locktime
        data += self.locktime.to_bytes(4, 'little')

        return data

    def serialize(self) -> bytes:
        """
        Serialize complete transaction.

        Returns:
            Complete transaction bytes
        """
        data = b''

        # Version
        data += self.version.to_bytes(4, 'little')

        # Inputs
        data += len(self.inputs).to_bytes(1, 'big')
        for inp in self.inputs:
            data += inp.prev_tx
            data += inp.output_index.to_bytes(4, 'little')
            data += len(inp.signature).to_bytes(1, 'big')
            data += inp.signature

        # Outputs
        data += len(self.outputs).to_bytes(1, 'big')
        for out in self.outputs:
            data += out.amount.to_bytes(8, 'little')
            data += out.address

        # Locktime
        data += self.locktime.to_bytes(4, 'little')

        return data

    @property
    def hash(self) -> bytes:
        """Transaction hash (Blake2b-256)."""
        return hashlib.blake2b(self.serialize(), digest_size=32).digest()

    @property
    def txid(self) -> str:
        """Transaction ID (hash as hex string)."""
        return self.hash.hex()

    def calculate_fee(self, input_amounts: List[int]) -> int:
        """
        Calculate transaction fee.

        Args:
            input_amounts: List of input amounts (from UTXOs)

        Returns:
            Fee amount (inputs - outputs)
        """
        total_in = sum(input_amounts)
        total_out = sum(out.amount for out in self.outputs)
        return total_in - total_out


class TransactionSigner:
    """
    Signs transactions with wallet keys.

    Supports:
    - Single signature
    - Multi-signature (future)
    - Hardware wallet integration (future)
    """

    def __init__(self, keypair: KeyPair):
        """
        Initialize signer with key pair.

        Args:
            keypair: Key pair for signing
        """
        self.keypair = keypair

    def sign_transaction(self, tx: Transaction) -> Transaction:
        """
        Sign all inputs in transaction.

        Assumes all inputs belong to this wallet.

        Args:
            tx: Transaction to sign

        Returns:
            Transaction with signatures filled in
        """
        # Get signing message
        message = tx.serialize_for_signing()

        # Sign each input
        signed_inputs = []
        for inp in tx.inputs:
            signature = self.keypair.sign(message)
            signed_inputs.append(TxInput(
                prev_tx=inp.prev_tx,
                output_index=inp.output_index,
                signature=signature
            ))

        return Transaction(
            version=tx.version,
            inputs=signed_inputs,
            outputs=tx.outputs,
            locktime=tx.locktime
        )

    def sign_input(
        self,
        tx: Transaction,
        input_index: int
    ) -> bytes:
        """
        Sign a specific input.

        For multi-party transactions where each signer
        controls different inputs.

        Args:
            tx: Transaction
            input_index: Index of input to sign

        Returns:
            64-byte signature
        """
        if input_index >= len(tx.inputs):
            raise ValueError(f"Invalid input index: {input_index}")

        message = tx.serialize_for_signing()
        return self.keypair.sign(message)

    def verify_signature(
        self,
        tx: Transaction,
        input_index: int,
        public_key: bytes
    ) -> bool:
        """
        Verify a signature on specific input.

        Args:
            tx: Transaction
            input_index: Input index
            public_key: Public key to verify with

        Returns:
            True if signature is valid
        """
        if input_index >= len(tx.inputs):
            return False

        message = tx.serialize_for_signing()
        signature = tx.inputs[input_index].signature

        return KeyPair.verify(public_key, message, signature)


@dataclass
class UTXO:
    """Unspent transaction output."""
    txid: bytes  # Transaction hash
    index: int  # Output index
    amount: int  # Amount
    address: bytes  # Owner address


@dataclass
class TransactionBuilder:
    """
    Builds transactions from UTXO set.

    Handles:
    - Input selection
    - Change output
    - Fee calculation
    """
    utxos: List[UTXO]  # Available UTXOs
    keypair: KeyPair
    fee_rate: int = 1  # satoshis per byte

    def select_inputs(self, amount: int) -> Tuple[List[UTXO], int]:
        """
        Select inputs for transaction.

        Uses simple greedy algorithm: largest first.

        Args:
            amount: Target amount (including fee)

        Returns:
            (selected_utxos, total_amount)
        """
        # Sort UTXOs by amount (largest first)
        sorted_utxos = sorted(self.utxos, key=lambda u: u.amount, reverse=True)

        selected = []
        total = 0

        for utxo in sorted_utxos:
            selected.append(utxo)
            total += utxo.amount

            if total >= amount:
                break

        return selected, total

    def estimate_fee(self, n_inputs: int, n_outputs: int) -> int:
        """
        Estimate transaction fee.

        Args:
            n_inputs: Number of inputs
            n_outputs: Number of outputs

        Returns:
            Estimated fee in smallest units
        """
        # Rough size estimation
        # Each input: ~100 bytes (32 txid + 4 index + 64 sig)
        # Each output: ~40 bytes (8 amount + 20 address)
        # Overhead: ~20 bytes
        size = 20 + (n_inputs * 100) + (n_outputs * 40)
        return size * self.fee_rate

    def build(
        self,
        recipients: List[Tuple[str, int]],  # (address, amount)
        change_address: Optional[str] = None
    ) -> Optional[Transaction]:
        """
        Build a transaction.

        Args:
            recipients: List of (address, amount) tuples
            change_address: Address for change (default: own address)

        Returns:
            Unsigned transaction, or None if insufficient funds
        """
        # Calculate total needed
        total_out = sum(amount for _, amount in recipients)

        # Estimate fee (initial guess)
        estimated_fee = self.estimate_fee(
            n_inputs=len(self.utxos),  # Worst case
            n_outputs=len(recipients) + 1  # Include change
        )

        # Select inputs
        selected_utxos, total_in = self.select_inputs(total_out + estimated_fee)

        if total_in < total_out + estimated_fee:
            return None  # Insufficient funds

        # Refine fee estimate with actual input count
        actual_fee = self.estimate_fee(
            n_inputs=len(selected_utxos),
            n_outputs=len(recipients) + 1
        )

        # Check if we still have enough
        if total_in < total_out + actual_fee:
            return None

        # Create inputs
        inputs = []
        for utxo in selected_utxos:
            inputs.append(TxInput(
                prev_tx=utxo.txid,
                output_index=utxo.index,
                signature=b'\x00' * 64  # Placeholder
            ))

        # Create outputs
        outputs = []
        for address, amount in recipients:
            outputs.append(TxOutput(
                amount=amount,
                address=address_to_bytes(address)
            ))

        # Change output
        change = total_in - total_out - actual_fee
        if change > 0:
            if not change_address:
                # Use own address for change
                change_address = public_key_to_address(self.keypair.public_key)

            outputs.append(TxOutput(
                amount=change,
                address=address_to_bytes(change_address)
            ))

        return Transaction(
            version=1,
            inputs=inputs,
            outputs=outputs,
            locktime=0
        )

    def build_and_sign(
        self,
        recipients: List[Tuple[str, int]],
        change_address: Optional[str] = None
    ) -> Optional[Transaction]:
        """
        Build and sign transaction in one step.

        Args:
            recipients: List of (address, amount) tuples
            change_address: Address for change

        Returns:
            Signed transaction, or None if insufficient funds
        """
        # Build unsigned transaction
        tx = self.build(recipients, change_address)
        if tx is None:
            return None

        # Sign it
        signer = TransactionSigner(self.keypair)
        return signer.sign_transaction(tx)


class MultiSigSigner:
    """
    Multi-signature transaction signing.

    Supports m-of-n multi-sig schemes.
    """

    def __init__(self, threshold: int, public_keys: List[bytes]):
        """
        Initialize multi-sig signer.

        Args:
            threshold: Number of signatures required
            public_keys: List of public keys
        """
        self.threshold = threshold
        self.public_keys = public_keys

        if threshold > len(public_keys):
            raise ValueError("Threshold cannot exceed number of keys")

    def create_multisig_address(self, testnet: bool = False) -> str:
        """
        Create multi-sig address.

        Uses hash of combined public keys.

        Args:
            testnet: Use testnet prefix

        Returns:
            Multi-sig address
        """
        # Combine public keys and threshold
        data = self.threshold.to_bytes(1, 'big')
        for pk in sorted(self.public_keys):  # Sort for deterministic address
            data += pk

        # Hash to get address
        addr_hash = hashlib.blake2b(data, digest_size=32).digest()[:20]

        # Create address with special multi-sig prefix
        from .address import base58_encode
        prefix = b'\x05' if not testnet else b'\xc4'  # Multi-sig prefixes
        versioned = prefix + addr_hash

        # Checksum
        checksum = hashlib.blake2b(
            hashlib.blake2b(versioned, digest_size=32).digest(),
            digest_size=32
        ).digest()[:4]

        return base58_encode(versioned + checksum)

    def verify_multisig(
        self,
        message: bytes,
        signatures: List[bytes]
    ) -> bool:
        """
        Verify multi-sig signatures.

        Args:
            message: Message that was signed
            signatures: List of signatures

        Returns:
            True if threshold is met
        """
        if len(signatures) < self.threshold:
            return False

        valid_count = 0

        for sig in signatures:
            # Try to verify with each public key
            for pk in self.public_keys:
                if KeyPair.verify(pk, message, sig):
                    valid_count += 1
                    break

        return valid_count >= self.threshold