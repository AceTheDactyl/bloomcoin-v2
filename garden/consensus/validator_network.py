"""
Validator network for consensus
"""

from typing import Dict, List, Any
from dataclasses import dataclass


@dataclass
class ValidatorNode:
    """A validator node in the network"""
    node_id: str
    reputation: float = 0.5
    validations_performed: int = 0
    successful_validations: int = 0


class ValidatorNetwork:
    """Network of validator agents"""

    def __init__(self):
        self.validators: Dict[str, ValidatorNode] = {}

    def register_validator(self, agent_id: str) -> ValidatorNode:
        """Register an agent as a validator"""
        if agent_id not in self.validators:
            self.validators[agent_id] = ValidatorNode(node_id=agent_id)
        return self.validators[agent_id]

    def select_validators(self, count: int, exclude: List[str] = None) -> List[str]:
        """Select validators for an event"""
        exclude = exclude or []
        available = [
            v_id for v_id in self.validators.keys()
            if v_id not in exclude
        ]

        # Sort by reputation
        available.sort(
            key=lambda x: self.validators[x].reputation,
            reverse=True
        )

        return available[:count]

    def update_reputation(self, validator_id: str, success: bool):
        """Update validator reputation"""
        if validator_id in self.validators:
            node = self.validators[validator_id]
            node.validations_performed += 1
            if success:
                node.successful_validations += 1
                # Increase reputation
                node.reputation = min(1.0, node.reputation + 0.01)
            else:
                # Decrease reputation
                node.reputation = max(0.0, node.reputation - 0.01)