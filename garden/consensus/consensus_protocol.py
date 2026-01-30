"""
Consensus Protocol for Garden - AI agents validating bloom events

Unlike traditional blockchain consensus (PoW/PoS), Garden uses Proof-of-Learning:
AI agents validate each other's learning through dialogue and verification.
"""

import time
import hashlib
import json
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
import logging


logger = logging.getLogger(__name__)


class ConsensusType(Enum):
    """Types of consensus mechanisms available"""
    PROOF_OF_LEARNING = "pol"       # AI validates by testing knowledge
    WITNESS_BASED = "witness"        # Direct witnesses confirm event
    COHERENCE_BASED = "coherence"    # Based on coherence with collective
    REPUTATION_BASED = "reputation"  # Weighted by agent reputation
    HYBRID = "hybrid"               # Combination of methods


@dataclass
class ConsensusRules:
    """
    Rules governing consensus in the Garden network.

    Based on golden ratio φ for harmonic thresholds.
    """
    min_validators: int = 1              # Minimum validators required
    max_validators: int = 7              # Maximum validators considered
    validation_timeout: float = 300.0    # Seconds to wait for validation
    coherence_threshold: float = 0.866   # z_c = √3/2 critical coherence
    reputation_weight: float = 0.618     # φ - 1 (golden ratio conjugate)
    consensus_threshold: float = 0.667   # 2/3 majority required
    challenge_difficulty: int = 3        # Difficulty of learning challenges

    def __post_init__(self):
        """Ensure thresholds follow golden ratio harmony"""
        phi = (1 + 5**0.5) / 2
        # Adjust thresholds to golden proportions if needed
        self.reputation_weight = 1 / phi  # φ^-1
        self.coherence_threshold = 3**0.5 / 2  # Critical coherence

    def required_validators(self, network_size: int) -> int:
        """
        Calculate required validators based on network size.

        Uses logarithmic scaling for efficiency.
        """
        if network_size < 3:
            return self.min_validators
        elif network_size < 10:
            return 2
        elif network_size < 50:
            return min(3, self.max_validators)
        else:
            # Logarithmic growth
            required = int(np.log2(network_size))
            return min(required, self.max_validators)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ConsensusProtocol:
    """
    Main consensus protocol for validating bloom events.

    Coordinates AI agents to reach agreement on the validity of learning events.
    """

    def __init__(
        self,
        consensus_type: ConsensusType = ConsensusType.HYBRID,
        rules: Optional[ConsensusRules] = None
    ):
        self.consensus_type = consensus_type
        self.rules = rules or ConsensusRules()

        # Validation pools for ongoing consensus
        self.validation_pools: Dict[str, ValidationPool] = {}

        # Agent reputation scores
        self.reputation_scores: Dict[str, float] = {}

        # Statistics
        self.stats = {
            "total_validations": 0,
            "successful_consensus": 0,
            "failed_consensus": 0,
            "average_time": 0.0,
            "challenges_issued": 0
        }

    def initiate_validation(
        self,
        event_id: str,
        event_data: Dict[str, Any],
        proposer_id: str,
        available_validators: List[str]
    ) -> 'ValidationPool':
        """
        Start the validation process for a bloom event.

        Args:
            event_id: Unique ID of the bloom event
            event_data: The event data to validate
            proposer_id: Agent proposing the event
            available_validators: List of available validator agent IDs

        Returns:
            ValidationPool managing this validation
        """
        # Select validators
        required = self.rules.required_validators(len(available_validators) + 1)
        validators = self._select_validators(
            available_validators,
            proposer_id,
            required
        )

        # Create validation pool
        pool = ValidationPool(
            event_id=event_id,
            event_data=event_data,
            proposer_id=proposer_id,
            validators=validators,
            rules=self.rules,
            start_time=time.time()
        )

        self.validation_pools[event_id] = pool
        self.stats["total_validations"] += 1

        logger.info(
            f"Validation initiated for event {event_id[:8]}... "
            f"with {len(validators)} validators"
        )

        return pool

    def _select_validators(
        self,
        available: List[str],
        exclude: str,
        count: int
    ) -> List[str]:
        """
        Select validators for an event.

        Prioritizes by reputation and diversity.
        """
        # Filter out the proposer
        candidates = [v for v in available if v != exclude]

        if len(candidates) <= count:
            return candidates

        # Sort by reputation (higher is better)
        candidates_with_rep = [
            (v, self.reputation_scores.get(v, 0.5))
            for v in candidates
        ]
        candidates_with_rep.sort(key=lambda x: x[1], reverse=True)

        # Take top by reputation but add some randomness for diversity
        top_validators = candidates_with_rep[:count*2]

        # Randomly select from top candidates
        if len(top_validators) > count:
            indices = np.random.choice(
                len(top_validators),
                size=count,
                replace=False
            )
            selected = [top_validators[i][0] for i in indices]
        else:
            selected = [v[0] for v in top_validators]

        return selected

    def submit_validation(
        self,
        event_id: str,
        validator_id: str,
        approved: bool,
        confidence: float = 1.0,
        evidence: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Submit a validation vote for an event.

        Args:
            event_id: The event being validated
            validator_id: ID of the validating agent
            approved: Whether the validator approves
            confidence: Confidence level (0-1)
            evidence: Supporting evidence for the decision

        Returns:
            True if consensus has been reached
        """
        if event_id not in self.validation_pools:
            logger.warning(f"No validation pool found for event {event_id}")
            return False

        pool = self.validation_pools[event_id]

        # Add the vote
        pool.add_vote(validator_id, approved, confidence, evidence)

        # Update validator reputation based on participation
        self._update_reputation(validator_id, 0.01)  # Small boost for participating

        # Check if consensus reached
        if pool.has_consensus():
            self._finalize_validation(event_id, pool)
            return True

        return False

    def _finalize_validation(
        self,
        event_id: str,
        pool: 'ValidationPool'
    ):
        """Finalize validation and update statistics"""
        consensus = pool.get_consensus_result()

        if consensus["approved"]:
            self.stats["successful_consensus"] += 1
            # Reward validators who agreed with consensus
            for vote in pool.votes.values():
                if vote["approved"] == consensus["approved"]:
                    self._update_reputation(vote["validator_id"], 0.05)
        else:
            self.stats["failed_consensus"] += 1

        # Update average time
        duration = time.time() - pool.start_time
        current_avg = self.stats["average_time"]
        n = self.stats["successful_consensus"] + self.stats["failed_consensus"]
        self.stats["average_time"] = (current_avg * (n-1) + duration) / n

        # Clean up
        del self.validation_pools[event_id]

        logger.info(
            f"Validation complete for {event_id[:8]}...: "
            f"{'APPROVED' if consensus['approved'] else 'REJECTED'} "
            f"(confidence: {consensus['confidence']:.2f})"
        )

    def _update_reputation(
        self,
        agent_id: str,
        delta: float
    ):
        """Update an agent's reputation score"""
        current = self.reputation_scores.get(agent_id, 0.5)
        # Use sigmoid to keep in range [0, 1]
        new_score = current + delta
        self.reputation_scores[agent_id] = 1 / (1 + np.exp(-new_score))

    def create_challenge(
        self,
        event_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Create a learning challenge to verify knowledge claim.

        Used in Proof-of-Learning consensus.
        """
        self.stats["challenges_issued"] += 1

        challenge_type = event_data.get("memory_type", "general")

        if challenge_type == "skill":
            # Challenge to demonstrate the skill
            return {
                "type": "demonstration",
                "request": "Please demonstrate the claimed skill",
                "evaluation_criteria": ["accuracy", "completeness"],
                "difficulty": self.rules.challenge_difficulty
            }
        elif challenge_type == "fact":
            # Question about the fact
            return {
                "type": "question",
                "request": "Answer questions about the learned fact",
                "questions": self._generate_questions(event_data),
                "difficulty": self.rules.challenge_difficulty
            }
        elif challenge_type == "creation":
            # Request to show the creation
            return {
                "type": "presentation",
                "request": "Present your creation for evaluation",
                "evaluation_criteria": ["originality", "quality"],
                "difficulty": self.rules.challenge_difficulty
            }
        else:
            # Generic challenge
            return {
                "type": "verification",
                "request": "Provide evidence of your learning",
                "difficulty": self.rules.challenge_difficulty
            }

    def _generate_questions(
        self,
        event_data: Dict[str, Any]
    ) -> List[str]:
        """Generate questions to test knowledge (simplified)"""
        # In production, use NLG to create relevant questions
        return [
            "What is the main concept you learned?",
            "How does this relate to your existing knowledge?",
            "Can you provide an example or application?"
        ]

    def get_statistics(self) -> Dict[str, Any]:
        """Get consensus protocol statistics"""
        return {
            **self.stats,
            "active_validations": len(self.validation_pools),
            "registered_validators": len(self.reputation_scores),
            "average_reputation": np.mean(list(self.reputation_scores.values()))
                                if self.reputation_scores else 0.5
        }


@dataclass
class ValidationPool:
    """
    Manages validation votes for a single bloom event.

    Tracks votes and determines when consensus is reached.
    """
    event_id: str
    event_data: Dict[str, Any]
    proposer_id: str
    validators: List[str]
    rules: ConsensusRules
    start_time: float
    votes: Dict[str, Dict[str, Any]] = None

    def __post_init__(self):
        if self.votes is None:
            self.votes = {}

    def add_vote(
        self,
        validator_id: str,
        approved: bool,
        confidence: float = 1.0,
        evidence: Optional[Dict[str, Any]] = None
    ):
        """Add a validation vote"""
        if validator_id not in self.validators:
            logger.warning(f"Validator {validator_id} not authorized for this event")
            return

        self.votes[validator_id] = {
            "validator_id": validator_id,
            "approved": approved,
            "confidence": confidence,
            "evidence": evidence,
            "timestamp": time.time()
        }

    def has_consensus(self) -> bool:
        """Check if consensus has been reached"""
        if len(self.votes) < len(self.validators) * 0.5:
            # Need at least half of validators
            return False

        # Check if timeout reached
        if time.time() - self.start_time > self.rules.validation_timeout:
            return True

        # Check if we have enough votes
        if len(self.votes) >= self.rules.min_validators:
            # Calculate approval rate
            approvals = sum(1 for v in self.votes.values() if v["approved"])
            approval_rate = approvals / len(self.votes)

            # Check against threshold
            return approval_rate >= self.rules.consensus_threshold

        return False

    def get_consensus_result(self) -> Dict[str, Any]:
        """Get the consensus result"""
        if not self.votes:
            return {"approved": False, "confidence": 0.0, "reason": "No votes"}

        # Calculate weighted approval
        total_weight = 0
        weighted_approval = 0

        for vote in self.votes.values():
            weight = vote["confidence"]
            total_weight += weight
            if vote["approved"]:
                weighted_approval += weight

        if total_weight == 0:
            return {"approved": False, "confidence": 0.0, "reason": "No confident votes"}

        approval_score = weighted_approval / total_weight
        approved = approval_score >= self.rules.consensus_threshold

        return {
            "approved": approved,
            "confidence": approval_score,
            "total_votes": len(self.votes),
            "validators": list(self.votes.keys()),
            "timestamp": time.time(),
            "reason": "Consensus reached" if approved else "Insufficient approval"
        }

    def get_status(self) -> Dict[str, Any]:
        """Get current status of validation pool"""
        return {
            "event_id": self.event_id,
            "proposer": self.proposer_id,
            "validators": self.validators,
            "votes_received": len(self.votes),
            "votes_needed": self.rules.min_validators,
            "time_elapsed": time.time() - self.start_time,
            "has_consensus": self.has_consensus()
        }