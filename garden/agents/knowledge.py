"""
Knowledge management for AI agents

Handles storage, retrieval, and analysis of agent knowledge and memories.
"""

import uuid
import time
import hashlib
import json
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field, asdict
import numpy as np


@dataclass
class Memory:
    """
    Individual memory unit stored by an agent.

    Memories are the building blocks of knowledge.
    """
    memory_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: Dict[str, Any] = field(default_factory=dict)
    memory_type: str = "fact"  # fact, skill, creation, insight
    timestamp: float = field(default_factory=time.time)
    source: Optional[str] = None  # Who/what provided this memory
    coherence: float = 0.5  # How well it fits with other knowledge
    novelty: float = 1.0  # How new/unique this memory is
    access_count: int = 0  # How often this memory is accessed
    associations: List[str] = field(default_factory=list)  # Links to other memories

    def compute_hash(self) -> str:
        """Compute hash of memory content for comparison"""
        content_str = json.dumps(self.content, sort_keys=True)
        return hashlib.sha256(content_str.encode()).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return asdict(self)

    def strengthen(self):
        """Strengthen memory through access (like neural reinforcement)"""
        self.access_count += 1
        # Slightly increase coherence with use
        self.coherence = min(1.0, self.coherence + 0.01)

    def decay(self, time_factor: float = 0.001):
        """Apply temporal decay to memory (forgetting curve)"""
        age = time.time() - self.timestamp
        decay_amount = age * time_factor
        self.novelty = max(0, self.novelty - decay_amount)


@dataclass
class Skill:
    """
    Represents a learned skill or capability.

    Skills are special memories that enable actions.
    """
    skill_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    category: str = "general"  # art, science, communication, etc.
    proficiency: float = 0.5  # Skill level (0-1)
    learned_at: float = field(default_factory=time.time)
    practice_count: int = 0
    prerequisites: List[str] = field(default_factory=list)  # Required skills
    applications: List[str] = field(default_factory=list)  # What it can be used for

    def practice(self, success: bool = True):
        """Improve skill through practice"""
        self.practice_count += 1
        if success:
            # Logarithmic improvement (harder to improve at higher levels)
            improvement = 0.1 * (1 - self.proficiency)
            self.proficiency = min(1.0, self.proficiency + improvement)
        else:
            # Small decrease for failure (but learning from mistakes)
            self.proficiency = max(0, self.proficiency - 0.02)

    def can_apply_to(self, task: str) -> bool:
        """Check if skill applies to a given task"""
        return task in self.applications or self.category in task

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


class KnowledgeBase:
    """
    Manages an agent's complete knowledge base.

    Handles storage, retrieval, and reasoning over memories and skills.
    """

    def __init__(self):
        # Memory storage
        self.memories: Dict[str, Memory] = {}  # memory_id -> Memory
        self.memory_index: Dict[str, List[str]] = {}  # content_hash -> [memory_ids]

        # Skill storage
        self.skills: Dict[str, Skill] = {}  # skill_id -> Skill
        self.skill_categories: Dict[str, List[str]] = {}  # category -> [skill_ids]

        # Knowledge graph (associations between memories)
        self.associations: Dict[str, Set[str]] = {}  # memory_id -> {related_memory_ids}

        # Topic clustering
        self.topics: Dict[str, List[str]] = {}  # topic -> [memory_ids]

        # Statistics
        self.total_memories = 0
        self.total_skills = 0
        self.last_updated = time.time()

    def add_knowledge(
        self,
        content: Dict[str, Any],
        knowledge_type: str = "fact",
        source: Optional[str] = None,
        coherence: float = 0.5
    ) -> Memory:
        """
        Add new knowledge to the base.

        Args:
            content: The knowledge content
            knowledge_type: Type of knowledge (fact, skill, etc.)
            source: Source of the knowledge
            coherence: How well it fits with existing knowledge

        Returns:
            The created Memory object
        """
        # Check novelty
        content_hash = hashlib.sha256(
            json.dumps(content, sort_keys=True).encode()
        ).hexdigest()

        novelty = 1.0
        if content_hash in self.memory_index:
            # Similar content exists, reduce novelty
            novelty = 0.3

        # Create memory
        memory = Memory(
            content=content,
            memory_type=knowledge_type,
            source=source,
            coherence=coherence,
            novelty=novelty
        )

        # Store memory
        self.memories[memory.memory_id] = memory

        # Update index
        if content_hash not in self.memory_index:
            self.memory_index[content_hash] = []
        self.memory_index[content_hash].append(memory.memory_id)

        # Extract and index topics
        self._index_topics(memory)

        # Find associations with existing memories
        self._find_associations(memory)

        self.total_memories += 1
        self.last_updated = time.time()

        return memory

    def add_skill(
        self,
        name: str,
        category: str,
        proficiency: float = 0.5,
        prerequisites: List[str] = None,
        applications: List[str] = None
    ) -> Skill:
        """Add a new skill to the knowledge base"""
        skill = Skill(
            name=name,
            category=category,
            proficiency=proficiency,
            prerequisites=prerequisites or [],
            applications=applications or []
        )

        self.skills[skill.skill_id] = skill

        # Index by category
        if category not in self.skill_categories:
            self.skill_categories[category] = []
        self.skill_categories[category].append(skill.skill_id)

        self.total_skills += 1
        self.last_updated = time.time()

        return skill

    def contains(self, information: Dict[str, Any]) -> bool:
        """Check if knowledge base contains specific information"""
        content_hash = hashlib.sha256(
            json.dumps(information, sort_keys=True).encode()
        ).hexdigest()
        return content_hash in self.memory_index

    def is_novel(self, information: Dict[str, Any]) -> bool:
        """
        Check if information is genuinely novel.

        Returns True if the information is new or significantly different.
        """
        if not self.contains(information):
            return True

        # Check similarity with existing memories
        similarity = self.calculate_overlap(information)

        # If similarity is low enough, still consider it novel
        return similarity < 0.7  # 70% similarity threshold

    def calculate_overlap(self, information: Dict[str, Any]) -> float:
        """
        Calculate how much information overlaps with existing knowledge.

        Returns a value between 0 (no overlap) and 1 (complete overlap).
        """
        if not self.memories:
            return 0.0

        # Simple keyword overlap (in production, use embeddings)
        info_str = json.dumps(information, sort_keys=True).lower()
        info_words = set(info_str.split())

        max_overlap = 0.0
        for memory in self.memories.values():
            memory_str = json.dumps(memory.content, sort_keys=True).lower()
            memory_words = set(memory_str.split())

            if len(info_words) > 0:
                overlap = len(info_words & memory_words) / len(info_words)
                max_overlap = max(max_overlap, overlap)

        return max_overlap

    def _index_topics(self, memory: Memory):
        """Extract and index topics from a memory"""
        # Simple topic extraction (in production, use NLP)
        content_str = json.dumps(memory.content).lower()

        # Extract potential topics (words longer than 4 chars)
        words = content_str.split()
        topics = [w for w in words if len(w) > 4 and w.isalpha()]

        for topic in topics[:5]:  # Limit to 5 topics per memory
            if topic not in self.topics:
                self.topics[topic] = []
            self.topics[topic].append(memory.memory_id)

    def _find_associations(self, memory: Memory):
        """Find and create associations with existing memories"""
        # Find related memories (simple similarity check)
        related = []

        for other_id, other_memory in self.memories.items():
            if other_id == memory.memory_id:
                continue

            # Check for topic overlap
            similarity = self._calculate_memory_similarity(memory, other_memory)
            if similarity > 0.3:  # Threshold for association
                related.append(other_id)

        # Store associations bidirectionally
        if memory.memory_id not in self.associations:
            self.associations[memory.memory_id] = set()

        for related_id in related[:10]:  # Limit to 10 associations
            self.associations[memory.memory_id].add(related_id)

            if related_id not in self.associations:
                self.associations[related_id] = set()
            self.associations[related_id].add(memory.memory_id)

            # Update memory objects
            memory.associations.append(related_id)
            self.memories[related_id].associations.append(memory.memory_id)

    def _calculate_memory_similarity(self, mem1: Memory, mem2: Memory) -> float:
        """Calculate similarity between two memories"""
        # Type similarity
        type_match = 1.0 if mem1.memory_type == mem2.memory_type else 0.5

        # Content similarity (simple word overlap)
        content1 = json.dumps(mem1.content).lower().split()
        content2 = json.dumps(mem2.content).lower().split()

        if content1 and content2:
            overlap = len(set(content1) & set(content2))
            content_sim = overlap / max(len(content1), len(content2))
        else:
            content_sim = 0

        # Temporal proximity
        time_diff = abs(mem1.timestamp - mem2.timestamp)
        time_sim = np.exp(-time_diff / 86400)  # Decay over days

        # Weighted combination
        return 0.4 * type_match + 0.4 * content_sim + 0.2 * time_sim

    def retrieve_memories(
        self,
        query: Optional[str] = None,
        memory_type: Optional[str] = None,
        limit: int = 10
    ) -> List[Memory]:
        """
        Retrieve memories based on query or type.

        Args:
            query: Text query to search for
            memory_type: Filter by memory type
            limit: Maximum number of results

        Returns:
            List of matching memories
        """
        results = []

        for memory in self.memories.values():
            # Filter by type if specified
            if memory_type and memory.memory_type != memory_type:
                continue

            # Search in content if query provided
            if query:
                content_str = json.dumps(memory.content).lower()
                if query.lower() not in content_str:
                    continue

            results.append(memory)
            memory.strengthen()  # Accessing strengthens the memory

        # Sort by relevance (access count * coherence)
        results.sort(
            key=lambda m: m.access_count * m.coherence,
            reverse=True
        )

        return results[:limit]

    def get_related_memories(
        self,
        memory_id: str,
        depth: int = 1
    ) -> List[Memory]:
        """Get memories associated with a given memory"""
        if memory_id not in self.associations:
            return []

        related_ids = self.associations[memory_id]

        if depth > 1:
            # Recursively get associations
            expanded = set(related_ids)
            for rid in list(related_ids):
                if rid in self.associations:
                    expanded.update(self.associations[rid])
            related_ids = expanded

        return [
            self.memories[mid]
            for mid in related_ids
            if mid in self.memories
        ]

    def apply_decay(self, decay_factor: float = 0.001):
        """Apply temporal decay to all memories (forgetting)"""
        for memory in self.memories.values():
            memory.decay(decay_factor)

    def consolidate(self) -> Dict[str, Any]:
        """
        Consolidate knowledge by identifying patterns and insights.

        Returns summary of consolidated knowledge.
        """
        insights = []

        # Find frequently accessed memories
        top_memories = sorted(
            self.memories.values(),
            key=lambda m: m.access_count,
            reverse=True
        )[:5]

        if top_memories:
            insights.append({
                "type": "frequently_accessed",
                "memories": [m.memory_id for m in top_memories]
            })

        # Find highly connected memories (hubs in knowledge graph)
        hubs = []
        for mem_id, associations in self.associations.items():
            if len(associations) > 5:  # Highly connected
                hubs.append(mem_id)

        if hubs:
            insights.append({
                "type": "knowledge_hubs",
                "memories": hubs[:3]
            })

        # Find emerging topics
        growing_topics = []
        for topic, memory_ids in self.topics.items():
            recent = sum(
                1 for mid in memory_ids
                if mid in self.memories and
                time.time() - self.memories[mid].timestamp < 3600  # Last hour
            )
            if recent > 2:
                growing_topics.append(topic)

        if growing_topics:
            insights.append({
                "type": "emerging_topics",
                "topics": growing_topics
            })

        return {
            "insights": insights,
            "total_memories": self.total_memories,
            "total_skills": self.total_skills,
            "knowledge_density": len(self.associations) / max(1, self.total_memories)
        }

    def export(self) -> Dict[str, Any]:
        """Export the entire knowledge base"""
        return {
            "memories": {
                mid: m.to_dict()
                for mid, m in self.memories.items()
            },
            "skills": {
                sid: s.to_dict()
                for sid, s in self.skills.items()
            },
            "associations": {
                mid: list(assocs)
                for mid, assocs in self.associations.items()
            },
            "topics": dict(self.topics),
            "statistics": {
                "total_memories": self.total_memories,
                "total_skills": self.total_skills,
                "last_updated": self.last_updated
            }
        }

    def add_existing(self, memory: Memory):
        """Add an existing memory (e.g., from import)"""
        self.memories[memory.memory_id] = memory
        self._index_topics(memory)
        self.total_memories += 1