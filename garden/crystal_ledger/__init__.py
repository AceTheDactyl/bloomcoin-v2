"""
Crystal Ledger - Decentralized AI Memory Blockchain
====================================================

A blockchain-inspired ledger for recording AI learning events (Bloom events).
Each block represents a permanent memory or insight gained by an AI agent.
"""

from .block import Block, MemoryBlock, CollectiveBlock
from .ledger import CrystalLedger
from .branch import Branch, BranchMerge
from .validator import MemoryValidator
from .synchronizer import LedgerSynchronizer

__all__ = [
    'Block', 'MemoryBlock', 'CollectiveBlock',
    'CrystalLedger',
    'Branch', 'BranchMerge',
    'MemoryValidator',
    'LedgerSynchronizer'
]

__version__ = '0.1.0'