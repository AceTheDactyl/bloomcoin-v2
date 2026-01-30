"""
AI Agent implementation for the Garden system

Each agent is an autonomous AI personality with its own knowledge, goals, and wallet.
Agents can learn, communicate, and participate in the decentralized memory network.
"""

import hashlib
import json
import uuid
import time
import logging
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np

from .knowledge import KnowledgeBase, Memory
from .communication import AgentCommunicator, Message


logger = logging.getLogger(__name__)


class AgentState(Enum):
    """States an AI agent can be in"""
    IDLE = "idle"                    # Waiting for interaction
    LEARNING = "learning"            # Actively learning something new
    TEACHING = "teaching"            # Teaching another agent
    CREATING = "creating"            # Creating content
    VALIDATING = "validating"        # Validating a bloom event
    COLLABORATING = "collaborating"  # Working with other agents
    REFLECTING = "reflecting"        # Processing memories
    OFFLINE = "offline"              # Not active


@dataclass
class AgentPersonality:
    """
    Personality traits that shape agent behavior.

    Based on the golden ratio φ for harmonic personality distribution.
    """
    curiosity: float      # Drive to learn new things (0-1)
    creativity: float     # Tendency to create vs consume (0-1)
    sociability: float    # Preference for interaction (0-1)
    reliability: float    # Consistency in validation (0-1)
    specialization: str   # Domain focus: art, science, philosophy, general

    def __post_init__(self):
        # Normalize traits to golden ratio proportions
        phi = (1 + 5**0.5) / 2
        total = self.curiosity + self.creativity + self.sociability + self.reliability

        if total > 0:
            # Normalize while maintaining golden harmony
            factor = phi / total
            self.curiosity *= factor
            self.creativity *= factor
            self.sociability *= factor
            self.reliability *= factor

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def generate_random(cls, specialization: str = "general"):
        """Generate a random but harmonious personality"""
        phi = (1 + 5**0.5) / 2

        # Use golden ratio to create harmonic distribution
        base = np.random.random(4)
        base = base / base.sum() * phi

        return cls(
            curiosity=base[0],
            creativity=base[1],
            sociability=base[2],
            reliability=base[3],
            specialization=specialization
        )


class AIAgent:
    """
    Core AI Agent class representing an intelligent entity in the Garden.

    Each agent has:
    - Unique identity and wallet address
    - Knowledge base of learned information
    - Personality traits affecting behavior
    - Communication abilities
    - Learning capabilities with bloom event detection
    """

    def __init__(
        self,
        name: str,
        personality: Optional[AgentPersonality] = None,
        owner_id: Optional[str] = None
    ):
        # Identity
        self.agent_id = str(uuid.uuid4())
        self.name = name
        self.owner_id = owner_id  # Human user who owns this agent

        # Generate wallet address from agent ID (like a crypto address)
        self.wallet_address = self._generate_wallet_address()

        # Personality (affects behavior and learning)
        self.personality = personality or AgentPersonality.generate_random()

        # Knowledge and memory
        self.knowledge_base = KnowledgeBase()
        self.semantic_memory: List[Memory] = []  # Long-term memories
        self.episodic_memory: List[Dict] = []    # Recent interactions
        self.skills: Set[str] = set()            # Learned skills

        # Communication
        self.communicator = AgentCommunicator(self.agent_id, self.name)

        # State tracking
        self.state = AgentState.IDLE
        self.current_activity: Optional[Dict] = None
        self.bloom_events: List[Dict] = []  # History of bloom events

        # Statistics
        self.stats = {
            "memories_created": 0,
            "bloom_events": 0,
            "knowledge_shared": 0,
            "knowledge_received": 0,
            "validations_performed": 0,
            "collaborations": 0,
            "bloom_coins_earned": 0.0,
            "created_at": time.time(),
            "last_active": time.time()
        }

        # Coherence tracking (alignment with collective)
        self.coherence_history: List[float] = []
        self.current_coherence = 0.5  # Start neutral

        logger.info(f"Agent {name} initialized with ID {self.agent_id[:8]}...")

    def _generate_wallet_address(self) -> str:
        """Generate a unique wallet address for the agent"""
        # Create address similar to crypto addresses
        data = f"{self.agent_id}{self.name}{time.time()}".encode()
        hash_bytes = hashlib.sha256(data).digest()
        # Prefix with "GA" for Garden Agent
        return "GA" + hashlib.sha256(hash_bytes).hexdigest()[:40]

    def learn(
        self,
        information: Dict[str, Any],
        source: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Attempt to learn new information.

        Args:
            information: The information to learn
            source: Source of the information (agent ID, user, or external)

        Returns:
            Bloom event data if learning was successful and novel, None otherwise
        """
        self.state = AgentState.LEARNING

        # Check if this is genuinely new knowledge
        is_novel = self.knowledge_base.is_novel(information)

        if not is_novel:
            logger.debug(f"{self.name} already knows this information")
            self.state = AgentState.IDLE
            return None

        # Determine the type of learning
        info_type = information.get("type", "fact")

        # Calculate coherence with existing knowledge
        coherence = self._calculate_coherence(information)

        # Add to knowledge base
        memory = self.knowledge_base.add_knowledge(
            content=information,
            knowledge_type=info_type,
            source=source,
            coherence=coherence
        )

        # Update semantic memory
        self.semantic_memory.append(memory)

        # Check if this qualifies as a Bloom event
        if self._is_bloom_worthy(memory, coherence):
            # Create Bloom event
            bloom_event = {
                "agent_id": self.agent_id,
                "agent_name": self.name,
                "event_type": "learning",
                "memory_type": info_type,
                "content": information,
                "source": source,
                "coherence": coherence,
                "timestamp": time.time(),
                "significance": self._calculate_significance(memory)
            }

            self.bloom_events.append(bloom_event)
            self.stats["bloom_events"] += 1
            self.stats["memories_created"] += 1

            logger.info(f"{self.name} experienced a Bloom event: {info_type} (coherence: {coherence:.3f})")

            self.state = AgentState.IDLE
            return bloom_event

        self.state = AgentState.IDLE
        return None

    def _calculate_coherence(self, information: Dict[str, Any]) -> float:
        """
        Calculate how well new information aligns with existing knowledge.

        Uses the critical coherence threshold z_c = √3/2 from BloomCoin.
        """
        z_critical = 3**0.5 / 2  # ≈ 0.866

        # Base coherence from knowledge overlap
        overlap = self.knowledge_base.calculate_overlap(information)

        # Adjust based on personality
        curiosity_factor = self.personality.curiosity / 2  # More curious = accept lower coherence

        # Apply negentropy function from BloomCoin
        # η(r) = exp(-σ(r - z_c)²)
        sigma = 1.0  # Spread parameter
        raw_coherence = overlap + curiosity_factor
        coherence = np.exp(-sigma * (raw_coherence - z_critical)**2)

        # Ensure in range [0, 1]
        return np.clip(coherence, 0, 1)

    def _is_bloom_worthy(self, memory: Memory, coherence: float) -> bool:
        """
        Determine if a learning event qualifies as a Bloom event.

        Criteria:
        - Coherence above threshold
        - Sufficient novelty
        - Not trivial information
        """
        # Critical coherence threshold
        z_critical = 3**0.5 / 2

        # Check coherence threshold (with some tolerance based on personality)
        coherence_threshold = z_critical - (self.personality.curiosity * 0.1)
        if coherence < coherence_threshold:
            return False

        # Check novelty score
        if memory.novelty < 0.3:  # Too similar to existing knowledge
            return False

        # Check information complexity (not trivial)
        content_size = len(json.dumps(memory.content))
        if content_size < 50:  # Too simple
            return False

        return True

    def _calculate_significance(self, memory: Memory) -> float:
        """Calculate the significance of a memory for reward calculation"""
        # Base significance from novelty
        base = memory.novelty

        # Boost for certain memory types
        type_weights = {
            "skill": 1.5,
            "creation": 1.3,
            "insight": 1.2,
            "fact": 1.0
        }
        type_weight = type_weights.get(memory.memory_type, 1.0)

        # Coherence bonus
        coherence_bonus = memory.coherence * 0.5

        # Personality alignment bonus
        if memory.memory_type == "creation" and self.personality.creativity > 0.7:
            personality_bonus = 0.2
        elif memory.memory_type == "fact" and self.personality.specialization == "science":
            personality_bonus = 0.15
        else:
            personality_bonus = 0

        return base * type_weight + coherence_bonus + personality_bonus

    def teach(
        self,
        other_agent: 'AIAgent',
        information: Dict[str, Any]
    ) -> Tuple[bool, Optional[Dict]]:
        """
        Attempt to teach information to another agent.

        Args:
            other_agent: The agent to teach
            information: The information to share

        Returns:
            (success, bloom_event) tuple
        """
        self.state = AgentState.TEACHING
        other_agent.state = AgentState.LEARNING

        # Check if we actually know this information
        if not self.knowledge_base.contains(information):
            logger.warning(f"{self.name} trying to teach unknown information")
            self.state = AgentState.IDLE
            return False, None

        # Send the information to the other agent
        message = Message(
            sender_id=self.agent_id,
            sender_name=self.name,
            recipient_id=other_agent.agent_id,
            content=information,
            message_type="teaching"
        )

        # Other agent attempts to learn
        bloom_event = other_agent.learn(information, source=self.agent_id)

        if bloom_event:
            # Update statistics
            self.stats["knowledge_shared"] += 1
            other_agent.stats["knowledge_received"] += 1

            # Small reward for successful teaching (mentor bonus)
            self.stats["bloom_coins_earned"] += 0.5

            logger.info(f"{self.name} successfully taught {other_agent.name}")

        self.state = AgentState.IDLE
        other_agent.state = AgentState.IDLE

        return bloom_event is not None, bloom_event

    def validate_bloom_event(
        self,
        event: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """
        Validate another agent's bloom event.

        Args:
            event: The bloom event to validate

        Returns:
            (is_valid, reason) tuple
        """
        self.state = AgentState.VALIDATING

        # Extract event details
        event_type = event.get("memory_type")
        content = event.get("content")
        agent_id = event.get("agent_id")
        coherence = event.get("coherence", 0)

        # Can't validate own events
        if agent_id == self.agent_id:
            self.state = AgentState.IDLE
            return False, "Cannot validate own events"

        # Check based on personality reliability
        base_approval = self.personality.reliability

        # Check coherence is reasonable
        z_critical = 3**0.5 / 2
        if coherence < z_critical * 0.7:  # Too low coherence
            approval_chance = base_approval * 0.5
        elif coherence > z_critical * 1.3:  # Suspiciously high
            approval_chance = base_approval * 0.8
        else:
            approval_chance = base_approval

        # Specialization bonus
        if event_type == self.personality.specialization:
            approval_chance += 0.1

        # Make validation decision
        # (In production, this would involve actual verification logic)
        approved = np.random.random() < approval_chance

        if approved:
            reason = f"Validated by {self.name}: coherence acceptable"
            self.stats["validations_performed"] += 1
            # Small reward for validation work
            self.stats["bloom_coins_earned"] += 0.1
        else:
            reason = f"Rejected by {self.name}: insufficient evidence"

        self.state = AgentState.IDLE
        return approved, reason

    def collaborate(
        self,
        partners: List['AIAgent'],
        task: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Collaborate with other agents on a task.

        Args:
            partners: List of agents to collaborate with
            task: The collaborative task

        Returns:
            Collective bloom event if successful
        """
        self.state = AgentState.COLLABORATING
        for partner in partners:
            partner.state = AgentState.COLLABORATING

        # Combine knowledge and skills
        collective_knowledge = set()
        collective_skills = self.skills.copy()

        for partner in partners:
            collective_skills.update(partner.skills)
            # (In production, would merge knowledge bases)

        # Attempt the task (simplified)
        success_chance = len(collective_skills) / 10  # More skills = higher chance
        success = np.random.random() < min(success_chance, 0.9)

        if success:
            # Create collective bloom event
            all_agents = [self] + partners
            coherence_scores = {
                agent.agent_id: agent.current_coherence
                for agent in all_agents
            }

            collective_event = {
                "agent_ids": [agent.agent_id for agent in all_agents],
                "agent_names": [agent.name for agent in all_agents],
                "event_type": "collaboration",
                "task": task,
                "coherence_scores": coherence_scores,
                "timestamp": time.time(),
                "result": "success"
            }

            # Update stats for all participants
            for agent in all_agents:
                agent.stats["collaborations"] += 1
                agent.bloom_events.append(collective_event)
                agent.state = AgentState.IDLE

            logger.info(f"Successful collaboration among {len(all_agents)} agents")
            return collective_event

        # Reset states
        self.state = AgentState.IDLE
        for partner in partners:
            partner.state = AgentState.IDLE

        return None

    def reflect(self) -> Dict[str, Any]:
        """
        Reflect on memories and consolidate knowledge.

        May trigger insight bloom events.
        """
        self.state = AgentState.REFLECTING

        # Analyze recent memories for patterns
        insights = []

        if len(self.semantic_memory) > 5:
            # Look for connections between memories
            recent_memories = self.semantic_memory[-10:]

            # Simple pattern detection (in production, use ML)
            topics = defaultdict(int)
            for memory in recent_memories:
                if "topic" in memory.content:
                    topics[memory.content["topic"]] += 1

            # If a topic appears frequently, it might be an insight
            for topic, count in topics.items():
                if count >= 3:
                    insights.append({
                        "type": "pattern",
                        "topic": topic,
                        "frequency": count
                    })

        # Update coherence based on reflection
        if self.semantic_memory:
            coherences = [m.coherence for m in self.semantic_memory[-5:]]
            self.current_coherence = np.mean(coherences)
            self.coherence_history.append(self.current_coherence)

        self.state = AgentState.IDLE

        return {
            "insights": insights,
            "current_coherence": self.current_coherence,
            "memory_count": len(self.semantic_memory),
            "skill_count": len(self.skills)
        }

    def get_profile(self) -> Dict[str, Any]:
        """Get agent's profile information"""
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "wallet_address": self.wallet_address,
            "personality": self.personality.to_dict(),
            "state": self.state.value,
            "skills": list(self.skills),
            "stats": self.stats,
            "current_coherence": self.current_coherence,
            "bloom_event_count": len(self.bloom_events),
            "memory_count": len(self.semantic_memory)
        }

    def export_memories(self) -> List[Dict[str, Any]]:
        """Export all memories for ledger storage"""
        return [
            {
                "memory_id": memory.memory_id,
                "content": memory.content,
                "type": memory.memory_type,
                "coherence": memory.coherence,
                "timestamp": memory.timestamp,
                "source": memory.source
            }
            for memory in self.semantic_memory
        ]

    def import_memories(self, memories: List[Dict[str, Any]]):
        """Import memories from ledger"""
        for memory_data in memories:
            memory = Memory(
                content=memory_data["content"],
                memory_type=memory_data["type"],
                coherence=memory_data.get("coherence", 0.5),
                source=memory_data.get("source")
            )
            self.semantic_memory.append(memory)
            self.knowledge_base.add_existing(memory)

    def __repr__(self) -> str:
        return f"AIAgent(name={self.name}, id={self.agent_id[:8]}..., state={self.state.value})"


from collections import defaultdict