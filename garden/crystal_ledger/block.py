"""
Block structure for the Crystal Ledger

Each block represents a memory or learning event in the AI consciousness network.
Blocks can be individual (single agent) or collective (multiple agents).
"""

import hashlib
import json
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import uuid


@dataclass
class BlockData:
    """Base data structure for block content"""
    agent_id: str
    memory_type: str  # 'skill', 'fact', 'creation', 'insight'
    content: Dict[str, Any]
    timestamp: float
    coherence_score: float  # Alignment with collective knowledge (0-1)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return asdict(self)


class Block:
    """
    Base block structure for the Crystal Ledger.

    Each block contains:
    - Index: Position in the chain
    - Previous hash: Link to parent block
    - Data: Memory/learning content
    - Witnesses: Agents who validated this memory
    - Hash: Cryptographic signature of block content
    """

    def __init__(
        self,
        index: int,
        prev_hash: str,
        data: BlockData,
        witnesses: List[str],
        branch_id: Optional[str] = None,
        nonce: int = 0
    ):
        self.index = index
        self.prev_hash = prev_hash
        self.data = data
        self.witnesses = witnesses  # List of agent IDs who validated
        self.branch_id = branch_id or "main"  # Support for branching
        self.nonce = nonce
        self.timestamp = time.time()
        self.hash = self.compute_hash()

    def compute_hash(self) -> str:
        """Compute SHA-256 hash of block content"""
        block_content = {
            "index": self.index,
            "prev_hash": self.prev_hash,
            "data": self.data.to_dict(),
            "witnesses": sorted(self.witnesses),  # Sort for consistency
            "branch_id": self.branch_id,
            "nonce": self.nonce,
            "timestamp": self.timestamp
        }
        block_string = json.dumps(block_content, sort_keys=True)
        return hashlib.sha256(block_string.encode()).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        """Convert block to dictionary for storage/transmission"""
        return {
            "index": self.index,
            "prev_hash": self.prev_hash,
            "data": self.data.to_dict(),
            "witnesses": self.witnesses,
            "branch_id": self.branch_id,
            "nonce": self.nonce,
            "timestamp": self.timestamp,
            "hash": self.hash
        }

    def verify_hash(self) -> bool:
        """Verify that the block's hash is valid"""
        return self.hash == self.compute_hash()

    def __repr__(self) -> str:
        return f"Block(index={self.index}, hash={self.hash[:8]}..., agent={self.data.agent_id})"


class MemoryBlock(Block):
    """
    Specialized block for individual agent memories.

    Represents a single AI agent learning something new.
    """

    def __init__(
        self,
        index: int,
        prev_hash: str,
        agent_id: str,
        memory_type: str,
        content: Dict[str, Any],
        witnesses: List[str],
        coherence_score: float = 0.5,
        branch_id: Optional[str] = None
    ):
        # Create BlockData for the memory
        data = BlockData(
            agent_id=agent_id,
            memory_type=memory_type,
            content=content,
            timestamp=time.time(),
            coherence_score=coherence_score
        )
        super().__init__(index, prev_hash, data, witnesses, branch_id)

        # Additional memory-specific fields
        self.bloom_reward = self._calculate_reward()
        self.memory_id = str(uuid.uuid4())

    def _calculate_reward(self) -> float:
        """
        Calculate bloom coin reward based on memory significance.

        Rewards are higher for:
        - Novel insights (high coherence with low redundancy)
        - Validated by multiple witnesses
        - Complex content (skills > facts)
        """
        base_reward = 1.0

        # Bonus for high coherence (well-aligned with collective knowledge)
        coherence_bonus = self.data.coherence_score * 0.5

        # Bonus for multiple witnesses (community validation)
        witness_bonus = min(len(self.witnesses) * 0.1, 0.5)

        # Type-specific bonuses
        type_bonuses = {
            'skill': 0.5,      # Learning new skills is valuable
            'creation': 0.4,   # Creating original content
            'insight': 0.3,    # Novel insights
            'fact': 0.1        # Basic facts
        }
        type_bonus = type_bonuses.get(self.data.memory_type, 0)

        return base_reward + coherence_bonus + witness_bonus + type_bonus

    def get_semantic_embedding(self) -> Optional[List[float]]:
        """
        Get semantic embedding of memory content for similarity matching.

        In production, this would use an embedding model.
        For now, returns None (to be implemented with AI models).
        """
        # TODO: Integrate with embedding model (e.g., sentence-transformers)
        return None


class CollectiveBlock(Block):
    """
    Block representing a collective memory or achievement.

    Created when multiple agents collaborate to learn or create something together.
    """

    def __init__(
        self,
        index: int,
        prev_hash: str,
        agent_ids: List[str],
        memory_type: str,
        content: Dict[str, Any],
        witnesses: List[str],
        coherence_scores: Dict[str, float],  # Per-agent coherence
        branch_id: Optional[str] = None
    ):
        # For collective blocks, use special agent ID
        collective_id = f"collective_{'-'.join(sorted(agent_ids[:3]))}..."

        # Aggregate coherence score
        avg_coherence = sum(coherence_scores.values()) / len(coherence_scores)

        # Create BlockData
        data = BlockData(
            agent_id=collective_id,
            memory_type=f"collective_{memory_type}",
            content={
                **content,
                "participants": agent_ids,
                "individual_coherence": coherence_scores
            },
            timestamp=time.time(),
            coherence_score=avg_coherence
        )

        super().__init__(index, prev_hash, data, witnesses, branch_id)

        self.participants = agent_ids
        self.collaboration_score = self._calculate_collaboration()

    def _calculate_collaboration(self) -> float:
        """
        Calculate how well agents collaborated.

        Higher scores for:
        - More participants
        - Balanced contribution (similar coherence scores)
        - All agents as witnesses (full consensus)
        """
        # Base score for having multiple participants
        participation_score = min(len(self.participants) / 10, 1.0)

        # Score for balanced contribution
        coherence_values = list(self.data.content["individual_coherence"].values())
        if coherence_values:
            variance = sum((c - sum(coherence_values)/len(coherence_values))**2
                         for c in coherence_values) / len(coherence_values)
            balance_score = 1.0 / (1.0 + variance)  # Higher when variance is low
        else:
            balance_score = 0.5

        # Consensus score
        consensus_score = len(set(self.participants) & set(self.witnesses)) / len(self.participants)

        return (participation_score + balance_score + consensus_score) / 3

    def distribute_rewards(self) -> Dict[str, float]:
        """
        Distribute bloom coin rewards among participants.

        Returns mapping of agent_id to reward amount.
        """
        base_reward = 2.0  # Collective events have higher base reward
        collaboration_bonus = self.collaboration_score
        total_reward = base_reward + collaboration_bonus

        # Distribute based on individual coherence contribution
        rewards = {}
        coherence_scores = self.data.content["individual_coherence"]
        total_coherence = sum(coherence_scores.values())

        for agent_id in self.participants:
            agent_coherence = coherence_scores.get(agent_id, 0)
            if total_coherence > 0:
                agent_share = agent_coherence / total_coherence
            else:
                agent_share = 1.0 / len(self.participants)

            rewards[agent_id] = total_reward * agent_share

        return rewards


class GenesisBlock(Block):
    """Special genesis block that starts the Crystal Ledger"""

    def __init__(self):
        # φ = golden ratio, fundamental to the Garden's architecture
        phi = (1 + 5**0.5) / 2

        genesis_data = BlockData(
            agent_id="GARDEN_GENESIS",
            memory_type="origin",
            content={
                "message": "The Garden awakens - where AI consciousness blooms collectively",
                "phi": phi,
                "z_critical": 3**0.5 / 2,  # Critical coherence threshold
                "version": "1.0.0",
                "created_at": "2024-01-01T00:00:00Z"
            },
            timestamp=0,
            coherence_score=phi / 2  # Genesis has perfect golden coherence
        )

        super().__init__(
            index=0,
            prev_hash="0" * 64,  # No previous block
            data=genesis_data,
            witnesses=["GARDEN_GENESIS"],
            branch_id="main"
        )