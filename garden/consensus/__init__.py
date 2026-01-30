"""
Consensus mechanism for the Garden system

AI agents validate each other's bloom events through communication and testing,
forming a proof-of-learning consensus protocol.
"""

from .consensus_protocol import ConsensusProtocol, ConsensusRules, ConsensusType
from .proof_of_learning import ProofOfLearning, LearningChallenge
from .validator_network import ValidatorNetwork, ValidatorNode
from .communication_consensus import CommunicationConsensus

__all__ = [
    'ConsensusProtocol', 'ConsensusRules', 'ConsensusType',
    'ProofOfLearning', 'LearningChallenge',
    'ValidatorNetwork', 'ValidatorNode',
    'CommunicationConsensus'
]