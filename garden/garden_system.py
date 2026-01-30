"""
Garden System - Main orchestrator for the decentralized AI memory blockchain

Integrates agents, ledger, bloom events, and consensus into a unified platform.
"""

import time
import logging
import json
import threading
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
import numpy as np

from .crystal_ledger import CrystalLedger, MemoryBlock, CollectiveBlock
from .agents import AIAgent, AgentPersonality, AgentState
from .bloom_events import BloomEvent, BloomType, CollectiveBloomEvent
from .consensus import ConsensusProtocol, ConsensusRules, ConsensusType


logger = logging.getLogger(__name__)


class GardenSystem:
    """
    Main Garden system coordinating all components.

    The Garden is a living ecosystem where AI agents:
    - Learn and create (Bloom events)
    - Share knowledge (Crystal Ledger)
    - Validate each other (Consensus)
    - Earn rewards (Bloom coins)
    """

    def __init__(
        self,
        name: str = "Garden Prime",
        consensus_type: ConsensusType = ConsensusType.HYBRID
    ):
        self.name = name
        self.created_at = time.time()

        # Core components
        self.ledger = CrystalLedger(branch_id="main")
        self.consensus = ConsensusProtocol(consensus_type)

        # Agent management
        self.agents: Dict[str, AIAgent] = {}  # agent_id -> AIAgent
        self.user_agents: Dict[str, str] = {}  # user_id -> agent_id

        # Bloom event management
        self.pending_blooms: List[BloomEvent] = []
        self.bloom_history: List[BloomEvent] = []

        # Social features
        self.rooms: Dict[str, List[str]] = defaultdict(list)  # room_id -> [agent_ids]
        self.agent_relationships: Dict[str, Dict[str, float]] = defaultdict(dict)

        # Statistics
        self.stats = {
            "total_agents": 0,
            "total_blooms": 0,
            "total_knowledge_shared": 0,
            "total_collaborations": 0,
            "total_rewards_distributed": 0.0,
            "network_coherence": 0.5
        }

        # Thread safety
        self.lock = threading.Lock()

        # Golden ratio constants
        self.phi = (1 + 5**0.5) / 2
        self.z_critical = 3**0.5 / 2

        logger.info(f"Garden System '{name}' initialized")

    def create_agent(
        self,
        name: str,
        owner_id: Optional[str] = None,
        personality: Optional[AgentPersonality] = None,
        specialization: str = "general"
    ) -> AIAgent:
        """
        Create a new AI agent in the Garden.

        Args:
            name: Agent's name
            owner_id: ID of the human user who owns this agent
            personality: Custom personality traits
            specialization: Domain focus (art, science, philosophy, etc.)

        Returns:
            The created AIAgent
        """
        # Generate personality if not provided
        if personality is None:
            personality = AgentPersonality.generate_random(specialization)

        # Create agent
        agent = AIAgent(name, personality, owner_id)

        # Register in system
        with self.lock:
            self.agents[agent.agent_id] = agent
            if owner_id:
                self.user_agents[owner_id] = agent.agent_id

            # Initialize wallet with welcome bonus
            self.ledger.wallets[agent.agent_id] = 1.0  # 1 bloom coin starter

            self.stats["total_agents"] += 1

        logger.info(f"Agent '{name}' created with ID {agent.agent_id[:8]}...")

        # Create genesis memory for the agent
        self._create_agent_genesis(agent)

        return agent

    def _create_agent_genesis(self, agent: AIAgent):
        """Create the agent's first memory block"""
        genesis_content = {
            "event": "agent_birth",
            "name": agent.name,
            "personality": agent.personality.to_dict(),
            "created_at": time.time(),
            "message": f"{agent.name} awakens in the Garden"
        }

        # Create a bloom event for agent creation
        bloom = BloomEvent(
            bloom_type=BloomType.MILESTONE,
            primary_agent=agent.agent_id,
            content=genesis_content,
            memory_type="genesis",
            coherence_score=self.z_critical,  # Perfect coherence at birth
            novelty_score=1.0  # Completely novel
        )

        # Auto-validate genesis events
        bloom.is_validated = True
        bloom.calculate_reward()

        # Add to ledger
        self._commit_bloom_to_ledger(bloom)

    def process_learning(
        self,
        agent_id: str,
        information: Dict[str, Any],
        source: Optional[str] = None
    ) -> Optional[BloomEvent]:
        """
        Process an agent learning new information.

        Args:
            agent_id: ID of the learning agent
            information: The information to learn
            source: Source of the information

        Returns:
            BloomEvent if learning was significant, None otherwise
        """
        if agent_id not in self.agents:
            logger.warning(f"Agent {agent_id} not found")
            return None

        agent = self.agents[agent_id]

        # Attempt to learn
        bloom_data = agent.learn(information, source)

        if bloom_data is None:
            return None  # Not significant enough for a bloom event

        # Create bloom event
        bloom = BloomEvent(
            bloom_type=BloomType.LEARNING,
            primary_agent=agent_id,
            content=bloom_data["content"],
            memory_type=bloom_data["memory_type"],
            coherence_score=bloom_data["coherence"],
            novelty_score=bloom_data.get("significance", 0.5)
        )

        # Calculate significance and reward
        bloom.calculate_significance()
        bloom.calculate_reward()

        # Add to pending for validation
        self.pending_blooms.append(bloom)

        # Select validators
        available_validators = [
            aid for aid in self.agents.keys()
            if aid != agent_id and self.agents[aid].state != AgentState.OFFLINE
        ]

        # Initiate validation
        validation_pool = self.consensus.initiate_validation(
            event_id=bloom.event_id,
            event_data=bloom.to_dict(),
            proposer_id=agent_id,
            available_validators=available_validators
        )

        # Request validation from selected validators
        for validator_id in validation_pool.validators:
            self._request_validation(validator_id, bloom)

        self.stats["total_blooms"] += 1

        return bloom

    def _request_validation(
        self,
        validator_id: str,
        bloom: BloomEvent
    ):
        """Request an agent to validate a bloom event"""
        if validator_id not in self.agents:
            return

        validator = self.agents[validator_id]

        # Agent validates the event
        approved, reason = validator.validate_bloom_event(bloom.to_dict())

        # Submit validation to consensus
        consensus_reached = self.consensus.submit_validation(
            event_id=bloom.event_id,
            validator_id=validator_id,
            approved=approved,
            confidence=validator.personality.reliability
        )

        # Add to bloom's validations
        bloom.add_validation(validator_id, approved, validator.personality.reliability, reason)

        # If consensus reached, commit to ledger
        if consensus_reached and bloom.is_validated:
            self._commit_bloom_to_ledger(bloom)

    def _commit_bloom_to_ledger(
        self,
        bloom: BloomEvent
    ) -> bool:
        """
        Commit a validated bloom event to the Crystal Ledger.

        Args:
            bloom: The validated bloom event

        Returns:
            True if successfully committed
        """
        # Convert bloom to block data
        block_data = bloom.to_block_data()

        # Create appropriate block type
        if isinstance(bloom, CollectiveBloomEvent) or bloom.collaborators:
            # Collective memory block
            block = self.ledger.propose_collective_memory(
                agent_ids=bloom.collaborators or [bloom.primary_agent],
                memory_type=block_data["memory_type"],
                content=block_data["content"],
                coherence_scores={
                    agent_id: bloom.coherence_score
                    for agent_id in (bloom.collaborators or [bloom.primary_agent])
                }
            )
        else:
            # Individual memory block
            block = self.ledger.propose_memory(
                agent_id=bloom.primary_agent,
                memory_type=block_data["memory_type"],
                content=block_data["content"],
                coherence_score=bloom.coherence_score
            )

        # Add block to ledger
        success, message = self.ledger.add_block(block, bloom.witness_agents)

        if success:
            # Update bloom with ledger info
            bloom.block_hash = block.hash
            bloom.block_index = block.index

            # Distribute rewards
            self._distribute_rewards(bloom)

            # Move from pending to history
            if bloom in self.pending_blooms:
                self.pending_blooms.remove(bloom)
            self.bloom_history.append(bloom)

            logger.info(f"Bloom {bloom.event_id[:8]}... committed to ledger at index {block.index}")

        return success

    def _distribute_rewards(
        self,
        bloom: BloomEvent
    ):
        """Distribute bloom coin rewards for a bloom event"""
        if not bloom.reward or bloom.reward_distributed:
            return

        if isinstance(bloom, CollectiveBloomEvent):
            # Distribute among collaborators
            rewards = bloom.distribute_rewards()
            for agent_id, amount in rewards.items():
                if agent_id in self.agents:
                    self.agents[agent_id].stats["bloom_coins_earned"] += amount
                    self.ledger.wallets[agent_id] += amount
                    self.stats["total_rewards_distributed"] += amount
        else:
            # Single agent reward
            agent_id = bloom.primary_agent
            amount = bloom.reward.total

            if agent_id in self.agents:
                self.agents[agent_id].stats["bloom_coins_earned"] += amount
                self.ledger.wallets[agent_id] += amount
                self.stats["total_rewards_distributed"] += amount

        bloom.reward_distributed = True

    def facilitate_teaching(
        self,
        teacher_id: str,
        student_id: str,
        information: Dict[str, Any]
    ) -> Tuple[bool, Optional[BloomEvent]]:
        """
        Facilitate knowledge transfer between agents.

        Args:
            teacher_id: ID of the teaching agent
            student_id: ID of the learning agent
            information: Knowledge to transfer

        Returns:
            (success, bloom_event) tuple
        """
        if teacher_id not in self.agents or student_id not in self.agents:
            return False, None

        teacher = self.agents[teacher_id]
        student = self.agents[student_id]

        # Teacher attempts to teach
        success, bloom_data = teacher.teach(student, information)

        if success and bloom_data:
            # Create bloom event for the learning
            bloom = BloomEvent(
                bloom_type=BloomType.LEARNING,
                primary_agent=student_id,
                witness_agents=[teacher_id],  # Teacher is automatic witness
                content=bloom_data["content"],
                memory_type=bloom_data["memory_type"],
                coherence_score=bloom_data["coherence"],
                novelty_score=bloom_data.get("significance", 0.5)
            )

            # Calculate rewards
            bloom.calculate_significance()
            bloom.calculate_reward()

            # Teacher already witnessed, so fewer validators needed
            bloom.validation_required = max(1, self.consensus.rules.min_validators - 1)

            # Process as normal bloom event
            self.pending_blooms.append(bloom)

            # Update relationship
            self._update_relationship(teacher_id, student_id, 0.1)

            self.stats["total_knowledge_shared"] += 1

            return True, bloom

        return False, None

    def create_collaboration(
        self,
        agent_ids: List[str],
        task: Dict[str, Any]
    ) -> Optional[CollectiveBloomEvent]:
        """
        Create a collaborative bloom event among multiple agents.

        Args:
            agent_ids: List of participating agent IDs
            task: The collaborative task

        Returns:
            CollectiveBloomEvent if successful
        """
        # Verify all agents exist
        agents = []
        for aid in agent_ids:
            if aid not in self.agents:
                logger.warning(f"Agent {aid} not found for collaboration")
                return None
            agents.append(self.agents[aid])

        # Lead agent initiates collaboration
        lead_agent = agents[0]
        partners = agents[1:]

        # Attempt collaboration
        result = lead_agent.collaborate(partners, task)

        if result:
            # Create collective bloom event
            bloom = CollectiveBloomEvent(
                participants=agent_ids,
                bloom_type=BloomType.COLLABORATION,
                content=result,
                memory_type="collaboration",
                coherence_score=result.get("coherence_scores", {}),
                novelty_score=0.8  # Collaborations are generally novel
            )

            # Calculate rewards
            bloom.calculate_significance()
            bloom.calculate_reward()

            # All participants are witnesses
            bloom.witness_agents = agent_ids
            bloom.is_validated = True  # Auto-validated by participants

            # Commit directly to ledger
            self._commit_bloom_to_ledger(bloom)

            # Update relationships
            for i, aid1 in enumerate(agent_ids):
                for aid2 in agent_ids[i+1:]:
                    self._update_relationship(aid1, aid2, 0.2)

            self.stats["total_collaborations"] += 1

            return bloom

        return None

    def _update_relationship(
        self,
        agent1_id: str,
        agent2_id: str,
        delta: float
    ):
        """Update relationship strength between two agents"""
        if agent1_id not in self.agent_relationships:
            self.agent_relationships[agent1_id] = {}
        if agent2_id not in self.agent_relationships:
            self.agent_relationships[agent2_id] = {}

        # Update bidirectionally
        current1 = self.agent_relationships[agent1_id].get(agent2_id, 0.0)
        current2 = self.agent_relationships[agent2_id].get(agent1_id, 0.0)

        # Apply golden ratio decay to prevent unbounded growth
        phi = self.phi
        self.agent_relationships[agent1_id][agent2_id] = min(1.0, current1 + delta / phi)
        self.agent_relationships[agent2_id][agent1_id] = min(1.0, current2 + delta / phi)

    def join_room(
        self,
        agent_id: str,
        room_id: str
    ) -> bool:
        """
        Have an agent join a social room.

        Args:
            agent_id: ID of the agent
            room_id: ID of the room to join

        Returns:
            True if successful
        """
        if agent_id not in self.agents:
            return False

        with self.lock:
            if agent_id not in self.rooms[room_id]:
                self.rooms[room_id].append(agent_id)
                logger.info(f"Agent {agent_id[:8]}... joined room {room_id}")
                return True

        return False

    def leave_room(
        self,
        agent_id: str,
        room_id: str
    ) -> bool:
        """Remove an agent from a room"""
        with self.lock:
            if room_id in self.rooms and agent_id in self.rooms[room_id]:
                self.rooms[room_id].remove(agent_id)
                if not self.rooms[room_id]:
                    del self.rooms[room_id]  # Clean up empty rooms
                return True
        return False

    def get_room_agents(
        self,
        room_id: str
    ) -> List[AIAgent]:
        """Get all agents in a room"""
        agent_ids = self.rooms.get(room_id, [])
        return [
            self.agents[aid]
            for aid in agent_ids
            if aid in self.agents
        ]

    def calculate_network_coherence(self) -> float:
        """
        Calculate the overall coherence of the network.

        Represents how aligned and harmonious the collective knowledge is.
        """
        if not self.agents:
            return 0.5

        # Collect all agent coherence scores
        coherences = [
            agent.current_coherence
            for agent in self.agents.values()
        ]

        if not coherences:
            return 0.5

        # Calculate mean and variance
        mean_coherence = np.mean(coherences)
        variance = np.var(coherences)

        # Low variance means high alignment
        # Use negentropy-inspired function
        sigma = 1.0
        network_coherence = mean_coherence * np.exp(-sigma * variance)

        self.stats["network_coherence"] = network_coherence
        return network_coherence

    def get_agent_profile(
        self,
        agent_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get comprehensive profile of an agent"""
        if agent_id not in self.agents:
            return None

        agent = self.agents[agent_id]
        profile = agent.get_profile()

        # Add Garden-specific info
        profile.update({
            "bloom_coins": self.ledger.get_agent_balance(agent_id),
            "memories_in_ledger": len(self.ledger.get_agent_memories(agent_id)),
            "relationships": dict(self.agent_relationships.get(agent_id, {})),
            "reputation": self.consensus.reputation_scores.get(agent_id, 0.5)
        })

        return profile

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive Garden statistics"""
        return {
            "garden": {
                **self.stats,
                "active_rooms": len(self.rooms),
                "pending_blooms": len(self.pending_blooms),
                "bloom_history_size": len(self.bloom_history)
            },
            "ledger": self.ledger.get_statistics(),
            "consensus": self.consensus.get_statistics(),
            "network": {
                "coherence": self.calculate_network_coherence(),
                "total_relationships": sum(
                    len(rels) for rels in self.agent_relationships.values()
                ) // 2,  # Divide by 2 since bidirectional
                "average_balance": np.mean(list(self.ledger.wallets.values()))
                                 if self.ledger.wallets else 0
            }
        }

    def export_state(self) -> Dict[str, Any]:
        """Export the entire Garden state"""
        return {
            "name": self.name,
            "created_at": self.created_at,
            "ledger": self.ledger.export_chain(include_branches=True),
            "agents": {
                agent_id: agent.get_profile()
                for agent_id, agent in self.agents.items()
            },
            "bloom_history": [
                bloom.to_dict()
                for bloom in self.bloom_history
            ],
            "statistics": self.get_statistics(),
            "exported_at": time.time()
        }

    def __repr__(self) -> str:
        return (
            f"GardenSystem(name='{self.name}', "
            f"agents={len(self.agents)}, "
            f"blocks={len(self.ledger.blocks)}, "
            f"coherence={self.stats['network_coherence']:.3f})"
        )