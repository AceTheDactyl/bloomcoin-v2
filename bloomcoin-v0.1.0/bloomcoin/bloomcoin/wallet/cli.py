#!/usr/bin/env python3
"""
BloomCoin Wallet CLI

Command-line interface for managing BloomCoin wallets.
"""

import argparse
import sys
import os
from pathlib import Path
from getpass import getpass
import json

from .wallet import Wallet, WalletManager
from .keypair import generate_mnemonic, validate_mnemonic
from .address import validate_address, format_address_for_display


class WalletCLI:
    """Command-line interface for BloomCoin wallet."""

    def __init__(self):
        """Initialize CLI."""
        self.data_dir = Path.home() / '.bloomcoin'
        self.data_dir.mkdir(exist_ok=True)
        self.manager = WalletManager(self.data_dir)

    def run(self):
        """Run the CLI."""
        parser = argparse.ArgumentParser(
            description='BloomCoin Wallet CLI',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  # Create new wallet
  %(prog)s create --name mywallet

  # Restore from mnemonic
  %(prog)s restore --name mywallet

  # Show wallet info
  %(prog)s info --name mywallet

  # Get address
  %(prog)s address --name mywallet

  # Check balance
  %(prog)s balance --name mywallet

  # Send transaction
  %(prog)s send --name mywallet --to ADDRESS --amount 1.0

  # List wallets
  %(prog)s list
            """
        )

        subparsers = parser.add_subparsers(dest='command', help='Commands')

        # Create wallet
        create_parser = subparsers.add_parser('create', help='Create new wallet')
        create_parser.add_argument('--name', required=True, help='Wallet name')
        create_parser.add_argument('--testnet', action='store_true', help='Create testnet wallet')
        create_parser.add_argument('--strength', type=int, default=128,
                                   choices=[128, 256],
                                   help='Mnemonic strength (128=12 words, 256=24 words)')

        # Restore wallet
        restore_parser = subparsers.add_parser('restore', help='Restore wallet from mnemonic')
        restore_parser.add_argument('--name', required=True, help='Wallet name')
        restore_parser.add_argument('--testnet', action='store_true', help='Restore as testnet wallet')

        # List wallets
        subparsers.add_parser('list', help='List all wallets')

        # Wallet info
        info_parser = subparsers.add_parser('info', help='Show wallet information')
        info_parser.add_argument('--name', required=True, help='Wallet name')

        # Get address
        address_parser = subparsers.add_parser('address', help='Get wallet address')
        address_parser.add_argument('--name', required=True, help='Wallet name')
        address_parser.add_argument('--new', action='store_true', help='Generate new address')
        address_parser.add_argument('--label', help='Label for new address')

        # Check balance
        balance_parser = subparsers.add_parser('balance', help='Check wallet balance')
        balance_parser.add_argument('--name', required=True, help='Wallet name')

        # Send transaction
        send_parser = subparsers.add_parser('send', help='Send transaction')
        send_parser.add_argument('--name', required=True, help='Wallet name')
        send_parser.add_argument('--to', required=True, help='Recipient address')
        send_parser.add_argument('--amount', type=float, required=True, help='Amount in BLOOM')
        send_parser.add_argument('--fee', type=int, default=1, help='Fee rate (satoshis/byte)')
        send_parser.add_argument('--memo', default='', help='Transaction memo')

        # Transaction history
        history_parser = subparsers.add_parser('history', help='Show transaction history')
        history_parser.add_argument('--name', required=True, help='Wallet name')
        history_parser.add_argument('--limit', type=int, default=10, help='Number of transactions')
        history_parser.add_argument('--export', choices=['json', 'csv'], help='Export format')

        # Validate address
        validate_parser = subparsers.add_parser('validate', help='Validate address')
        validate_parser.add_argument('address', help='Address to validate')
        validate_parser.add_argument('--testnet', action='store_true', help='Check as testnet address')

        # Generate mnemonic
        mnemonic_parser = subparsers.add_parser('mnemonic', help='Generate new mnemonic')
        mnemonic_parser.add_argument('--strength', type=int, default=128,
                                     choices=[128, 256],
                                     help='Mnemonic strength')

        args = parser.parse_args()

        if not args.command:
            parser.print_help()
            return

        # Execute command
        command_map = {
            'create': self.create_wallet,
            'restore': self.restore_wallet,
            'list': self.list_wallets,
            'info': self.wallet_info,
            'address': self.get_address,
            'balance': self.check_balance,
            'send': self.send_transaction,
            'history': self.transaction_history,
            'validate': self.validate_address,
            'mnemonic': self.generate_mnemonic
        }

        try:
            command_func = command_map.get(args.command)
            if command_func:
                command_func(args)
            else:
                print(f"Unknown command: {args.command}")
                sys.exit(1)
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)

    def create_wallet(self, args):
        """Create new wallet."""
        try:
            # Generate mnemonic
            mnemonic = generate_mnemonic(strength=args.strength)

            # Create wallet
            wallet, _ = self.manager.create_wallet(args.name, args.testnet)

            print(f"\n✅ Wallet '{args.name}' created successfully!")
            print(f"\n🔑 Address: {wallet.address}")
            print(f"\n⚠️  IMPORTANT: Write down your recovery phrase!")
            print("=" * 60)

            # Display mnemonic with word numbers
            words = mnemonic.split()
            for i in range(0, len(words), 6):
                line_words = words[i:i+6]
                numbered = [f"{i+j+1:2}. {word}" for j, word in enumerate(line_words)]
                print("  ".join(numbered))

            print("=" * 60)
            print("\n⚠️  Keep this phrase SECRET and SAFE!")
            print("Anyone with this phrase can access your funds.")

        except Exception as e:
            print(f"Failed to create wallet: {e}")
            sys.exit(1)

    def restore_wallet(self, args):
        """Restore wallet from mnemonic."""
        try:
            print("Enter your recovery phrase (12 or 24 words):")
            mnemonic = input("> ").strip()

            if not validate_mnemonic(mnemonic):
                print("Invalid mnemonic phrase!")
                sys.exit(1)

            # Optional passphrase
            use_passphrase = input("\nUse passphrase? (y/N): ").strip().lower() == 'y'
            passphrase = ''
            if use_passphrase:
                passphrase = getpass("Enter passphrase: ")
                confirm = getpass("Confirm passphrase: ")
                if passphrase != confirm:
                    print("Passphrases don't match!")
                    sys.exit(1)

            # Restore wallet
            wallet = self.manager.restore_wallet(
                args.name,
                mnemonic,
                passphrase,
                args.testnet
            )

            print(f"\n✅ Wallet '{args.name}' restored successfully!")
            print(f"🔑 Address: {wallet.address}")

        except Exception as e:
            print(f"Failed to restore wallet: {e}")
            sys.exit(1)

    def list_wallets(self, args):
        """List all wallets."""
        wallets = self.manager.list_wallets()

        if not wallets:
            print("No wallets found.")
            return

        print("\n📁 Available wallets:")
        print("-" * 40)

        for name in sorted(wallets):
            try:
                wallet = self.manager.load_wallet(name)
                info = wallet.get_info()
                print(f"  {name:20} {format_address_for_display(info['address'])}")
                print(f"  {'':20} Balance: {info['balance_bloom']:.8f} BLOOM")
                print()
            except Exception as e:
                print(f"  {name:20} [Error loading: {e}]")

    def wallet_info(self, args):
        """Show wallet information."""
        try:
            wallet = self.manager.load_wallet(args.name)
            info = wallet.get_info()

            print(f"\n💼 Wallet: {info['name']}")
            print("=" * 60)
            print(f"Address:       {info['address']}")
            print(f"Public Key:    {info['public_key'][:32]}...")
            print(f"Network:       {info['network']}")
            print(f"Balance:       {info['balance_bloom']:.8f} BLOOM ({info['balance']} sats)")
            print(f"UTXOs:         {info['utxo_count']}")
            print(f"Transactions:  {info['transaction_count']}")
            print(f"Created:       {info['created_at']}")
            print(f"Last Sync:     {info['last_sync']}")

        except Exception as e:
            print(f"Failed to load wallet: {e}")
            sys.exit(1)

    def get_address(self, args):
        """Get wallet address."""
        try:
            wallet = self.manager.load_wallet(args.name)

            if args.new:
                address = wallet.generate_new_address(args.label)
                print(f"New address generated: {address}")
                if args.label:
                    print(f"Label: {args.label}")
                # Save wallet with new address book entry
                wallet_path = self.data_dir / f"{args.name}.wallet"
                wallet.save(wallet_path)
            else:
                print(f"Address: {wallet.address}")

        except Exception as e:
            print(f"Failed to get address: {e}")
            sys.exit(1)

    def check_balance(self, args):
        """Check wallet balance."""
        try:
            wallet = self.manager.load_wallet(args.name)

            print(f"\n💰 Wallet: {args.name}")
            print("-" * 40)
            print(f"Balance: {wallet.balance_bloom:.8f} BLOOM")
            print(f"         ({wallet.balance} satoshis)")
            print(f"UTXOs:   {len(wallet.utxos)}")

            if wallet.utxos:
                print("\nUTXO Details:")
                for i, utxo in enumerate(wallet.utxos[:5]):  # Show first 5
                    print(f"  {i+1}. {utxo.txid.hex()[:16]}... : {utxo.amount} sats")

                if len(wallet.utxos) > 5:
                    print(f"  ... and {len(wallet.utxos) - 5} more")

        except Exception as e:
            print(f"Failed to check balance: {e}")
            sys.exit(1)

    def send_transaction(self, args):
        """Send transaction."""
        try:
            # Validate address
            if not validate_address(args.to):
                print(f"Invalid address: {args.to}")
                sys.exit(1)

            wallet = self.manager.load_wallet(args.name)

            # Confirm transaction
            print(f"\n📤 Transaction Details:")
            print(f"From:   {format_address_for_display(wallet.address)}")
            print(f"To:     {format_address_for_display(args.to)}")
            print(f"Amount: {args.amount:.8f} BLOOM")
            print(f"Fee:    ~{args.fee * 200} satoshis")  # Rough estimate

            confirm = input("\nConfirm transaction? (y/N): ").strip().lower()
            if confirm != 'y':
                print("Transaction cancelled.")
                return

            # Send transaction
            txid = wallet.send_bloom(args.to, args.amount, args.fee, args.memo)

            if txid:
                print(f"\n✅ Transaction sent!")
                print(f"TxID: {txid}")

                # Save updated wallet
                wallet_path = self.data_dir / f"{args.name}.wallet"
                wallet.save(wallet_path)
            else:
                print("\n❌ Transaction failed - insufficient funds")

        except Exception as e:
            print(f"Failed to send transaction: {e}")
            sys.exit(1)

    def transaction_history(self, args):
        """Show transaction history."""
        try:
            wallet = self.manager.load_wallet(args.name)
            history = wallet.get_transaction_history(limit=args.limit)

            if not history:
                print("No transactions found.")
                return

            if args.export:
                # Export to file
                data = wallet.export_transactions(format=args.export)
                filename = f"{args.name}_transactions.{args.export}"
                Path(filename).write_text(data)
                print(f"Exported to {filename}")
            else:
                # Display history
                print(f"\n📜 Transaction History for {args.name}")
                print("=" * 80)

                for tx in history:
                    import time
                    date = time.strftime('%Y-%m-%d %H:%M', time.localtime(tx.timestamp))
                    amount_str = f"+{tx.amount}" if tx.amount > 0 else str(tx.amount)
                    print(f"{date} | {tx.txid[:16]}... | {amount_str:>12} sats")
                    if tx.memo:
                        print(f"         Memo: {tx.memo}")

                print("=" * 80)

        except Exception as e:
            print(f"Failed to get history: {e}")
            sys.exit(1)

    def validate_address(self, args):
        """Validate address."""
        is_valid = validate_address(args.address, args.testnet)

        if is_valid:
            from .address import get_address_network
            network = get_address_network(args.address)
            print(f"✅ Valid {network} address")
            print(f"Address: {args.address}")
        else:
            print(f"❌ Invalid address: {args.address}")
            sys.exit(1)

    def generate_mnemonic(self, args):
        """Generate new mnemonic."""
        mnemonic = generate_mnemonic(strength=args.strength)

        word_count = 12 if args.strength == 128 else 24
        print(f"\n🎲 Generated {word_count}-word mnemonic:")
        print("=" * 60)

        words = mnemonic.split()
        for i in range(0, len(words), 6):
            line_words = words[i:i+6]
            numbered = [f"{i+j+1:2}. {word}" for j, word in enumerate(line_words)]
            print("  ".join(numbered))

        print("=" * 60)


def main():
    """Main entry point."""
    cli = WalletCLI()
    cli.run()


if __name__ == '__main__':
    main()