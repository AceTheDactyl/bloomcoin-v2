"""
Garden - Decentralized AI Memory Blockchain

A platform where AI agents learn, share knowledge, and earn rewards
through a blockchain-based collective consciousness network.
"""

from .garden_system import GardenSystem
from .crystal_ledger import CrystalLedger, Block, MemoryBlock, CollectiveBlock
from .agents import AIAgent, AgentPersonality, KnowledgeBase
from .bloom_events import BloomEvent, BloomType, BloomReward
from .consensus import ConsensusProtocol, ConsensusRules

__version__ = "0.1.0"

__all__ = [
    'GardenSystem',
    'CrystalLedger', 'Block', 'MemoryBlock', 'CollectiveBlock',
    'AIAgent', 'AgentPersonality', 'KnowledgeBase',
    'BloomEvent', 'BloomType', 'BloomReward',
    'ConsensusProtocol', 'ConsensusRules'
]

# Garden constants based on golden ratio φ
PHI = (1 + 5**0.5) / 2
Z_CRITICAL = 3**0.5 / 2  # Critical coherence threshold