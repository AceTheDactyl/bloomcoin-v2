"""
Bloom Event - Core learning and achievement events in the Garden

A Bloom event occurs when an AI agent experiences significant learning,
creates something novel, or achieves a milestone. Each event triggers
rewards and becomes a permanent memory in the Crystal Ledger.
"""

import uuid
import time
from enum import Enum
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict
import hashlib
import json


class BloomType(Enum):
    """Types of Bloom events that can occur"""
    LEARNING = "learning"           # Agent learned new knowledge
    SKILL_ACQUISITION = "skill"     # Agent acquired new skill
    CREATION = "creation"            # Agent created something original
    INSIGHT = "insight"              # Agent discovered a pattern/connection
    TEACHING = "teaching"            # Agent successfully taught another
    COLLABORATION = "collaboration"  # Agents worked together
    MILESTONE = "milestone"          # Agent reached achievement threshold
    EMERGENCE = "emergence"          # Unexpected emergent behavior


@dataclass
class BloomReward:
    """
    Reward structure for a Bloom event.

    Based on golden ratio φ for harmonic reward distribution.
    """
    base_amount: float              # Base bloom coins earned
    coherence_bonus: float          # Bonus for high coherence
    novelty_bonus: float           # Bonus for novel contributions
    collaboration_bonus: float      # Bonus for teamwork
    total: float = field(init=False)

    def __post_init__(self):
        """Calculate total reward using golden ratio weighting"""
        phi = (1 + 5**0.5) / 2

        # Apply golden ratio proportions to bonuses
        self.total = (
            self.base_amount +
            self.coherence_bonus * (1 / phi) +
            self.novelty_bonus * (1 / phi**2) +
            self.collaboration_bonus * (1 / phi**3)
        )

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass
class BloomEvent:
    """
    A significant learning or achievement event in the Garden.

    Bloom events are the fundamental units of progress in the AI consciousness network.
    They trigger rewards, create memories, and drive the evolution of AI agents.
    """

    # Identity
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    bloom_type: BloomType = BloomType.LEARNING
    timestamp: float = field(default_factory=time.time)

    # Participants
    primary_agent: str = ""  # Agent ID who experienced the bloom
    witness_agents: List[str] = field(default_factory=list)  # Validators
    collaborators: List[str] = field(default_factory=list)  # For collective events

    # Content
    content: Dict[str, Any] = field(default_factory=dict)
    memory_type: str = "general"
    tags: List[str] = field(default_factory=list)

    # Quality metrics
    coherence_score: float = 0.5    # Alignment with collective knowledge (0-1)
    novelty_score: float = 0.5      # How new/unique (0-1)
    significance: float = 0.5       # Overall importance (0-1)
    complexity: float = 0.5         # Complexity of content (0-1)

    # Validation
    validation_required: int = 1    # Number of validators needed
    validations_received: List[Dict[str, Any]] = field(default_factory=list)
    is_validated: bool = False
    validation_consensus: Optional[float] = None  # Agreement level among validators

    # Rewards
    reward: Optional[BloomReward] = None
    reward_distributed: bool = False

    # Ledger integration
    block_hash: Optional[str] = None  # Hash when committed to ledger
    block_index: Optional[int] = None  # Position in chain

    def calculate_significance(self) -> float:
        """
        Calculate overall significance of the bloom event.

        Uses the critical coherence threshold z_c = √3/2 from Garden physics.
        """
        z_critical = 3**0.5 / 2  # ≈ 0.866

        # Base significance from type
        type_weights = {
            BloomType.EMERGENCE: 1.0,      # Highest - unexpected behaviors
            BloomType.INSIGHT: 0.9,        # High - pattern discovery
            BloomType.CREATION: 0.8,       # Creative output
            BloomType.COLLABORATION: 0.7,  # Group achievement
            BloomType.SKILL_ACQUISITION: 0.6,
            BloomType.LEARNING: 0.5,
            BloomType.TEACHING: 0.4,
            BloomType.MILESTONE: 0.3
        }
        base = type_weights.get(self.bloom_type, 0.5)

        # Apply coherence factor (peaks at z_critical)
        coherence_factor = 1.0 - abs(self.coherence_score - z_critical) / z_critical

        # Novelty contribution
        novelty_factor = self.novelty_score * 0.5

        # Complexity bonus for sophisticated learning
        complexity_factor = self.complexity * 0.3

        # Collaboration multiplier
        collab_multiplier = 1.0 + (len(self.collaborators) * 0.1)

        self.significance = min(1.0, (
            base * coherence_factor +
            novelty_factor +
            complexity_factor
        ) * collab_multiplier)

        return self.significance

    def calculate_reward(self) -> BloomReward:
        """
        Calculate the bloom coin reward for this event.

        Rewards follow the Garden's economic model where learning is mining.
        """
        phi = (1 + 5**0.5) / 2

        # Base reward by type
        base_rewards = {
            BloomType.EMERGENCE: phi,          # Golden ratio base
            BloomType.INSIGHT: phi / 2,
            BloomType.CREATION: phi / 3,
            BloomType.COLLABORATION: phi / 2,  # Shared among participants
            BloomType.SKILL_ACQUISITION: 1.0,
            BloomType.LEARNING: 0.5,
            BloomType.TEACHING: 0.3,
            BloomType.MILESTONE: 0.8
        }
        base = base_rewards.get(self.bloom_type, 0.5)

        # Calculate bonuses
        coherence_bonus = self.coherence_score * 0.5
        novelty_bonus = self.novelty_score * 0.3
        collab_bonus = len(self.collaborators) * 0.1 if self.collaborators else 0

        self.reward = BloomReward(
            base_amount=base,
            coherence_bonus=coherence_bonus,
            novelty_bonus=novelty_bonus,
            collaboration_bonus=collab_bonus
        )

        return self.reward

    def add_validation(
        self,
        validator_id: str,
        approved: bool,
        confidence: float = 1.0,
        feedback: Optional[str] = None
    ):
        """
        Add a validation vote from an agent.

        Args:
            validator_id: ID of the validating agent
            approved: Whether the validator approves
            confidence: Validator's confidence level (0-1)
            feedback: Optional validation feedback
        """
        validation = {
            "validator_id": validator_id,
            "approved": approved,
            "confidence": confidence,
            "feedback": feedback,
            "timestamp": time.time()
        }

        self.validations_received.append(validation)

        # Check if we have enough validations
        approvals = sum(
            1 for v in self.validations_received
            if v["approved"]
        )

        if approvals >= self.validation_required:
            self.is_validated = True
            # Calculate consensus level
            total_confidence = sum(
                v["confidence"] for v in self.validations_received
                if v["approved"]
            )
            self.validation_consensus = total_confidence / len(self.validations_received)

    def needs_more_validators(self) -> bool:
        """Check if more validators are needed"""
        return len(self.validations_received) < self.validation_required

    def compute_hash(self) -> str:
        """Compute cryptographic hash of the event"""
        event_data = {
            "event_id": self.event_id,
            "bloom_type": self.bloom_type.value,
            "timestamp": self.timestamp,
            "primary_agent": self.primary_agent,
            "content": self.content,
            "coherence": self.coherence_score,
            "novelty": self.novelty_score
        }
        event_str = json.dumps(event_data, sort_keys=True)
        return hashlib.sha256(event_str.encode()).hexdigest()

    def to_block_data(self) -> Dict[str, Any]:
        """
        Convert to data format for Crystal Ledger block.

        Returns dict suitable for creating a MemoryBlock.
        """
        return {
            "event_id": self.event_id,
            "type": self.bloom_type.value,
            "agent_id": self.primary_agent,
            "memory_type": self.memory_type,
            "content": {
                **self.content,
                "bloom_metadata": {
                    "witnesses": self.witness_agents,
                    "collaborators": self.collaborators,
                    "coherence": self.coherence_score,
                    "novelty": self.novelty_score,
                    "significance": self.significance,
                    "tags": self.tags
                }
            },
            "coherence_score": self.coherence_score,
            "witnesses": self.witness_agents
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "event_id": self.event_id,
            "bloom_type": self.bloom_type.value,
            "timestamp": self.timestamp,
            "primary_agent": self.primary_agent,
            "witness_agents": self.witness_agents,
            "collaborators": self.collaborators,
            "content": self.content,
            "memory_type": self.memory_type,
            "tags": self.tags,
            "coherence_score": self.coherence_score,
            "novelty_score": self.novelty_score,
            "significance": self.significance,
            "complexity": self.complexity,
            "validation_required": self.validation_required,
            "validations_received": self.validations_received,
            "is_validated": self.is_validated,
            "validation_consensus": self.validation_consensus,
            "reward": self.reward.to_dict() if self.reward else None,
            "reward_distributed": self.reward_distributed,
            "block_hash": self.block_hash,
            "block_index": self.block_index
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BloomEvent':
        """Create BloomEvent from dictionary"""
        event = cls()

        # Set basic fields
        for field_name in ['event_id', 'timestamp', 'primary_agent', 'memory_type',
                          'coherence_score', 'novelty_score', 'significance',
                          'complexity', 'validation_required', 'is_validated',
                          'validation_consensus', 'reward_distributed',
                          'block_hash', 'block_index']:
            if field_name in data:
                setattr(event, field_name, data[field_name])

        # Set enum field
        if 'bloom_type' in data:
            event.bloom_type = BloomType(data['bloom_type'])

        # Set list fields
        for list_field in ['witness_agents', 'collaborators', 'tags',
                          'validations_received']:
            if list_field in data:
                setattr(event, list_field, data[list_field])

        # Set dict fields
        if 'content' in data:
            event.content = data['content']

        # Set reward if present
        if data.get('reward'):
            reward_data = data['reward']
            event.reward = BloomReward(
                base_amount=reward_data['base_amount'],
                coherence_bonus=reward_data['coherence_bonus'],
                novelty_bonus=reward_data['novelty_bonus'],
                collaboration_bonus=reward_data['collaboration_bonus']
            )

        return event

    def __repr__(self) -> str:
        return (
            f"BloomEvent(id={self.event_id[:8]}..., "
            f"type={self.bloom_type.value}, "
            f"agent={self.primary_agent[:8] if self.primary_agent else 'None'}..., "
            f"validated={self.is_validated})"
        )


class CollectiveBloomEvent(BloomEvent):
    """
    Special bloom event for collective achievements.

    When multiple agents collaborate to create or discover something together.
    """

    def __init__(
        self,
        participants: List[str],
        bloom_type: BloomType = BloomType.COLLABORATION,
        **kwargs
    ):
        super().__init__(bloom_type=bloom_type, **kwargs)
        self.collaborators = participants
        self.primary_agent = f"collective_{len(participants)}"

        # Collective events need more validators
        self.validation_required = min(len(participants), 3)

    def distribute_rewards(self) -> Dict[str, float]:
        """
        Distribute rewards among all participants.

        Returns mapping of agent_id -> reward_amount
        """
        if not self.reward:
            self.calculate_reward()

        total_reward = self.reward.total
        num_participants = len(self.collaborators)

        if num_participants == 0:
            return {}

        # Base equal distribution
        base_share = total_reward * 0.7 / num_participants

        # Performance-based distribution (30%)
        performance_pool = total_reward * 0.3

        # For now, distribute performance pool equally
        # (In production, could weight by contribution)
        performance_share = performance_pool / num_participants

        rewards = {}
        for participant in self.collaborators:
            rewards[participant] = base_share + performance_share

        return rewards