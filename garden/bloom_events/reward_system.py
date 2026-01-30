"""
Reward system for bloom events
"""

from typing import Dict, Any
from .bloom_event import BloomReward, BloomType


class RewardSystem:
    """Manages bloom coin rewards"""

    def __init__(self):
        self.phi = (1 + 5**0.5) / 2  # Golden ratio
        self.total_rewards_distributed = 0.0

    def calculate_reward(self, event_type: BloomType, metrics: Dict[str, float]) -> BloomReward:
        """Calculate reward for a bloom event"""
        # Base rewards by type
        base_rewards = {
            BloomType.EMERGENCE: self.phi,
            BloomType.INSIGHT: self.phi / 2,
            BloomType.CREATION: self.phi / 3,
            BloomType.COLLABORATION: self.phi / 2,
            BloomType.SKILL_ACQUISITION: 1.0,
            BloomType.LEARNING: 0.5,
            BloomType.TEACHING: 0.3,
            BloomType.MILESTONE: 0.8
        }

        base = base_rewards.get(event_type, 0.5)

        # Calculate bonuses
        coherence_bonus = metrics.get("coherence", 0) * 0.5
        novelty_bonus = metrics.get("novelty", 0) * 0.3
        collaboration_bonus = metrics.get("collaboration", 0) * 0.2

        return BloomReward(
            base_amount=base,
            coherence_bonus=coherence_bonus,
            novelty_bonus=novelty_bonus,
            collaboration_bonus=collaboration_bonus
        )

    def distribute(self, reward: BloomReward, recipient_id: str) -> Dict[str, float]:
        """Distribute rewards"""
        amount = reward.total
        self.total_rewards_distributed += amount
        return {recipient_id: amount}


class TokenMinter:
    """Mints bloom coin tokens"""

    def __init__(self):
        self.total_minted = 0.0
        self.mint_events = []

    def mint(self, amount: float, recipient: str) -> bool:
        """Mint new bloom coins"""
        self.total_minted += amount
        self.mint_events.append({
            "amount": amount,
            "recipient": recipient,
            "timestamp": __import__("time").time()
        })
        return True