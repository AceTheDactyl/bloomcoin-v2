"""
BloomCoin Wallet

Complete wallet functionality for managing keys, addresses, and transactions.
"""

from dataclasses import dataclass, field
from pathlib import Path
import json
import time
from typing import Optional, List, Tuple, Dict
import logging

from .keypair import (
    KeyPair,
    generate_mnemonic,
    keypair_from_mnemonic,
    validate_mnemonic,
    secure_zero
)
from .address import (
    public_key_to_address,
    address_to_bytes,
    validate_address,
    get_address_network,
    AddressBook
)
from .signer import (
    TransactionSigner,
    TransactionBuilder,
    Transaction,
    TxInput,
    TxOutput,
    UTXO
)

logger = logging.getLogger(__name__)


@dataclass
class WalletTransaction:
    """Record of a wallet transaction."""
    txid: str
    timestamp: float
    amount: int  # Positive for received, negative for sent
    fee: int
    address: str  # Other party's address
    confirmations: int = 0
    memo: str = ""


@dataclass
class Wallet:
    """
    BloomCoin wallet.

    Manages:
    - Key pairs
    - Addresses
    - UTXO tracking
    - Transaction building and signing
    - Transaction history
    """
    name: str
    keypair: KeyPair
    utxos: List[UTXO] = field(default_factory=list)
    testnet: bool = False
    address_book: AddressBook = field(default_factory=AddressBook)
    transaction_history: List[WalletTransaction] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    last_sync: float = 0.0

    @property
    def address(self) -> str:
        """Primary receiving address."""
        return public_key_to_address(self.keypair.public_key, self.testnet)

    @property
    def public_key(self) -> bytes:
        """Wallet public key."""
        return self.keypair.public_key

    @property
    def balance(self) -> int:
        """Current balance in smallest units."""
        return sum(utxo.amount for utxo in self.utxos)

    @property
    def balance_bloom(self) -> float:
        """Current balance in BLOOM."""
        return self.balance / 100_000_000  # 1 BLOOM = 10^8 smallest units

    @classmethod
    def create(cls, name: str, testnet: bool = False) -> Tuple['Wallet', str]:
        """
        Create new wallet with mnemonic backup.

        Args:
            name: Wallet name
            testnet: Use testnet

        Returns:
            (wallet, mnemonic)
        """
        mnemonic = generate_mnemonic(strength=128)  # 12 words
        keypair = keypair_from_mnemonic(mnemonic)
        wallet = cls(name=name, keypair=keypair, testnet=testnet)
        logger.info(f"Created new wallet: {name}")
        return wallet, mnemonic

    @classmethod
    def restore(
        cls,
        name: str,
        mnemonic: str,
        passphrase: str = '',
        testnet: bool = False
    ) -> 'Wallet':
        """
        Restore wallet from mnemonic.

        Args:
            name: Wallet name
            mnemonic: BIP39 mnemonic phrase
            passphrase: Optional passphrase
            testnet: Use testnet

        Returns:
            Restored wallet
        """
        if not validate_mnemonic(mnemonic):
            raise ValueError("Invalid mnemonic phrase")

        keypair = keypair_from_mnemonic(mnemonic, passphrase)
        wallet = cls(name=name, keypair=keypair, testnet=testnet)
        logger.info(f"Restored wallet: {name}")
        return wallet

    def generate_new_address(self, label: Optional[str] = None) -> str:
        """
        Generate a new address (for HD wallets, future feature).

        For now, returns the single address.

        Args:
            label: Optional label for address book

        Returns:
            New address
        """
        # In future, implement BIP32 HD derivation
        # For now, return primary address
        address = self.address

        if label:
            self.address_book.add(label, address, self.testnet)

        return address

    def send(
        self,
        to_address: str,
        amount: int,
        fee_rate: int = 1,
        memo: str = ""
    ) -> Optional[str]:
        """
        Send funds to address.

        Args:
            to_address: Recipient address
            amount: Amount in smallest units
            fee_rate: Fee rate in satoshis per byte
            memo: Optional transaction memo

        Returns:
            Transaction ID if successful, None if insufficient funds
        """
        # Validate address
        if not validate_address(to_address):
            raise ValueError(f"Invalid address: {to_address}")

        # Check network match
        network = get_address_network(to_address)
        expected = 'testnet' if self.testnet else 'mainnet'
        if network != expected:
            raise ValueError(f"Address is for {network}, wallet is {expected}")

        # Build transaction
        builder = TransactionBuilder(
            utxos=self.utxos,
            keypair=self.keypair,
            fee_rate=fee_rate
        )

        tx = builder.build_and_sign([(to_address, amount)])
        if tx is None:
            logger.warning("Insufficient funds for transaction")
            return None

        # Calculate fee
        total_in = sum(utxo.amount for utxo in builder.utxos)
        total_out = sum(out.amount for out in tx.outputs)
        fee = total_in - total_out

        # Record in history
        self.transaction_history.append(WalletTransaction(
            txid=tx.txid,
            timestamp=time.time(),
            amount=-amount,  # Negative for sent
            fee=fee,
            address=to_address,
            memo=memo
        ))

        # Update UTXOs (remove spent ones)
        spent_utxos = set()
        for inp in tx.inputs:
            for utxo in self.utxos:
                if utxo.txid == inp.prev_tx and utxo.index == inp.output_index:
                    spent_utxos.add(utxo)

        self.utxos = [u for u in self.utxos if u not in spent_utxos]

        logger.info(f"Sent {amount} to {to_address}, tx: {tx.txid}")

        # In real implementation, broadcast to network
        # network.broadcast_tx(tx)

        return tx.txid

    def send_bloom(
        self,
        to_address: str,
        amount_bloom: float,
        fee_rate: int = 1,
        memo: str = ""
    ) -> Optional[str]:
        """
        Send funds in BLOOM units.

        Args:
            to_address: Recipient address
            amount_bloom: Amount in BLOOM
            fee_rate: Fee rate
            memo: Optional memo

        Returns:
            Transaction ID if successful
        """
        amount_sats = int(amount_bloom * 100_000_000)
        return self.send(to_address, amount_sats, fee_rate, memo)

    def receive(self, tx: Transaction):
        """
        Process received transaction.

        Args:
            tx: Received transaction
        """
        my_addr_bytes = address_to_bytes(self.address)

        # Check outputs for our address
        for i, output in enumerate(tx.outputs):
            if output.address == my_addr_bytes:
                # Add to UTXOs
                utxo = UTXO(
                    txid=tx.hash,
                    index=i,
                    amount=output.amount,
                    address=output.address
                )
                self.utxos.append(utxo)

                # Add to history
                self.transaction_history.append(WalletTransaction(
                    txid=tx.txid,
                    timestamp=time.time(),
                    amount=output.amount,  # Positive for received
                    fee=0,
                    address="unknown",  # Would need to track sender
                    memo=""
                ))

                logger.info(f"Received {output.amount} in tx {tx.txid}")

    def update_utxos_from_chain(self, chain):
        """
        Scan blockchain for wallet UTXOs.

        Args:
            chain: Blockchain instance
        """
        logger.info("Scanning blockchain for UTXOs...")

        my_addr_bytes = address_to_bytes(self.address)
        self.utxos.clear()
        spent = set()

        # Track all transactions
        for block in chain.iterate_blocks():
            for tx in block.transactions:
                # Mark spent outputs
                for inp in tx.inputs:
                    spent.add((inp.prev_tx, inp.output_index))

                # Add our outputs
                for i, out in enumerate(tx.outputs):
                    if out.address == my_addr_bytes:
                        utxo_key = (tx.hash, i)
                        if utxo_key not in spent:
                            self.utxos.append(UTXO(
                                txid=tx.hash,
                                index=i,
                                amount=out.amount,
                                address=out.address
                            ))

        self.last_sync = time.time()
        logger.info(f"Found {len(self.utxos)} UTXOs, balance: {self.balance_bloom} BLOOM")

    def export_transactions(self, format: str = 'json') -> str:
        """
        Export transaction history.

        Args:
            format: Export format ('json' or 'csv')

        Returns:
            Exported data string
        """
        if format == 'json':
            data = []
            for tx in self.transaction_history:
                data.append({
                    'txid': tx.txid,
                    'timestamp': tx.timestamp,
                    'date': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(tx.timestamp)),
                    'amount': tx.amount,
                    'fee': tx.fee,
                    'address': tx.address,
                    'confirmations': tx.confirmations,
                    'memo': tx.memo
                })
            return json.dumps(data, indent=2)

        elif format == 'csv':
            lines = ['Date,TxID,Amount,Fee,Address,Memo']
            for tx in self.transaction_history:
                date = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(tx.timestamp))
                lines.append(f'{date},{tx.txid},{tx.amount},{tx.fee},{tx.address},{tx.memo}')
            return '\n'.join(lines)

        else:
            raise ValueError(f"Unsupported format: {format}")

    def get_transaction_history(
        self,
        limit: Optional[int] = None,
        offset: int = 0
    ) -> List[WalletTransaction]:
        """
        Get transaction history.

        Args:
            limit: Maximum transactions to return
            offset: Skip first N transactions

        Returns:
            List of transactions
        """
        history = sorted(self.transaction_history, key=lambda t: t.timestamp, reverse=True)

        if limit:
            return history[offset:offset+limit]
        else:
            return history[offset:]

    def save(self, path: Path, password: Optional[str] = None):
        """
        Save wallet to file.

        WARNING: In production, always encrypt with strong password!

        Args:
            path: File path
            password: Encryption password (required in production)
        """
        data = {
            'version': 1,
            'name': self.name,
            'private_key': self.keypair.private_key.hex(),
            'testnet': self.testnet,
            'created_at': self.created_at,
            'last_sync': self.last_sync,
            'address_book': self.address_book.export_json(),
            'transaction_history': [
                {
                    'txid': tx.txid,
                    'timestamp': tx.timestamp,
                    'amount': tx.amount,
                    'fee': tx.fee,
                    'address': tx.address,
                    'confirmations': tx.confirmations,
                    'memo': tx.memo
                }
                for tx in self.transaction_history
            ]
        }

        if password:
            # In production, encrypt data with password using AES-256-GCM
            # For now, just add a warning flag
            data['encrypted'] = False
            data['warning'] = 'Encryption not implemented - DO NOT USE IN PRODUCTION'

        path.write_text(json.dumps(data, indent=2))
        logger.info(f"Saved wallet to {path}")

    @classmethod
    def load(cls, path: Path, password: Optional[str] = None) -> 'Wallet':
        """
        Load wallet from file.

        Args:
            path: File path
            password: Decryption password

        Returns:
            Loaded wallet
        """
        data = json.loads(path.read_text())

        if data.get('encrypted'):
            if not password:
                raise ValueError("Password required for encrypted wallet")
            # In production, decrypt with password
            raise NotImplementedError("Encryption not yet implemented")

        # Restore wallet
        keypair = KeyPair.from_private_key(bytes.fromhex(data['private_key']))
        wallet = cls(
            name=data['name'],
            keypair=keypair,
            testnet=data.get('testnet', False),
            created_at=data.get('created_at', time.time()),
            last_sync=data.get('last_sync', 0.0)
        )

        # Restore address book
        if 'address_book' in data:
            wallet.address_book.import_json(data['address_book'])

        # Restore transaction history
        if 'transaction_history' in data:
            for tx_data in data['transaction_history']:
                wallet.transaction_history.append(WalletTransaction(
                    txid=tx_data['txid'],
                    timestamp=tx_data['timestamp'],
                    amount=tx_data['amount'],
                    fee=tx_data['fee'],
                    address=tx_data['address'],
                    confirmations=tx_data.get('confirmations', 0),
                    memo=tx_data.get('memo', '')
                ))

        logger.info(f"Loaded wallet from {path}")
        return wallet

    def get_info(self) -> Dict:
        """
        Get wallet information.

        Returns:
            Wallet info dictionary
        """
        return {
            'name': self.name,
            'address': self.address,
            'public_key': self.keypair.public_key.hex(),
            'network': 'testnet' if self.testnet else 'mainnet',
            'balance': self.balance,
            'balance_bloom': self.balance_bloom,
            'utxo_count': len(self.utxos),
            'transaction_count': len(self.transaction_history),
            'created_at': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.created_at)),
            'last_sync': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.last_sync)) if self.last_sync else 'Never'
        }

    def __repr__(self) -> str:
        return f"Wallet(name='{self.name}', address='{self.address[:10]}...', balance={self.balance_bloom:.8f} BLOOM)"


class WalletManager:
    """
    Manages multiple wallets.
    """

    def __init__(self, data_dir: Path):
        """
        Initialize wallet manager.

        Args:
            data_dir: Directory for wallet storage
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.wallets: Dict[str, Wallet] = {}

    def create_wallet(self, name: str, testnet: bool = False) -> Tuple[Wallet, str]:
        """
        Create new wallet.

        Args:
            name: Wallet name
            testnet: Use testnet

        Returns:
            (wallet, mnemonic)
        """
        if name in self.wallets:
            raise ValueError(f"Wallet '{name}' already exists")

        wallet, mnemonic = Wallet.create(name, testnet)
        self.wallets[name] = wallet

        # Save to file
        wallet_path = self.data_dir / f"{name}.wallet"
        wallet.save(wallet_path)

        return wallet, mnemonic

    def restore_wallet(
        self,
        name: str,
        mnemonic: str,
        passphrase: str = '',
        testnet: bool = False
    ) -> Wallet:
        """
        Restore wallet from mnemonic.

        Args:
            name: Wallet name
            mnemonic: BIP39 mnemonic
            passphrase: Optional passphrase
            testnet: Use testnet

        Returns:
            Restored wallet
        """
        if name in self.wallets:
            raise ValueError(f"Wallet '{name}' already exists")

        wallet = Wallet.restore(name, mnemonic, passphrase, testnet)
        self.wallets[name] = wallet

        # Save to file
        wallet_path = self.data_dir / f"{name}.wallet"
        wallet.save(wallet_path)

        return wallet

    def load_wallet(self, name: str, password: Optional[str] = None) -> Wallet:
        """
        Load wallet from file.

        Args:
            name: Wallet name
            password: Decryption password

        Returns:
            Loaded wallet
        """
        if name in self.wallets:
            return self.wallets[name]

        wallet_path = self.data_dir / f"{name}.wallet"
        if not wallet_path.exists():
            raise ValueError(f"Wallet '{name}' not found")

        wallet = Wallet.load(wallet_path, password)
        self.wallets[name] = wallet

        return wallet

    def list_wallets(self) -> List[str]:
        """
        List available wallets.

        Returns:
            List of wallet names
        """
        wallet_files = self.data_dir.glob("*.wallet")
        return [f.stem for f in wallet_files]

    def get_wallet(self, name: str) -> Optional[Wallet]:
        """
        Get loaded wallet.

        Args:
            name: Wallet name

        Returns:
            Wallet if loaded, None otherwise
        """
        return self.wallets.get(name)