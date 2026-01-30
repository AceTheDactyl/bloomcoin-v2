"""
Crystal Ledger - The decentralized memory blockchain for AI agents

Manages the chain of memory blocks, validation, and synchronization.
"""

import json
import threading
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
import time
import logging

from .block import Block, MemoryBlock, CollectiveBlock, GenesisBlock, BlockData


logger = logging.getLogger(__name__)


class CrystalLedger:
    """
    Main ledger class managing the blockchain of AI memories.

    Features:
    - Immutable append-only structure
    - Support for branching and merging
    - Distributed consensus among AI agents
    - Bloom event tracking and rewards
    """

    def __init__(self, branch_id: str = "main"):
        self.branch_id = branch_id
        self.blocks: List[Block] = []
        self.pending_blocks: List[Block] = []
        self.branches: Dict[str, List[Block]] = defaultdict(list)

        # Agent wallets (bloom coin balances)
        self.wallets: Dict[str, float] = defaultdict(float)

        # Memory index for quick lookup
        self.memory_index: Dict[str, Block] = {}  # memory_id -> block
        self.agent_memories: Dict[str, List[Block]] = defaultdict(list)  # agent -> blocks

        # Consensus tracking
        self.validation_pool: Dict[str, List[str]] = {}  # block_hash -> validators

        # Thread safety
        self.lock = threading.Lock()

        # Statistics
        self.stats = {
            "total_blocks": 0,
            "total_blooms": 0,
            "total_rewards": 0.0,
            "collective_events": 0,
            "branch_merges": 0
        }

        # Initialize with genesis block
        self._initialize_genesis()

    def _initialize_genesis(self):
        """Create and add the genesis block"""
        genesis = GenesisBlock()
        self.blocks.append(genesis)
        self.stats["total_blocks"] = 1
        logger.info(f"Crystal Ledger initialized with genesis block: {genesis.hash[:8]}...")

    def add_block(
        self,
        block: Block,
        validators: Optional[List[str]] = None
    ) -> Tuple[bool, str]:
        """
        Add a new block to the ledger after validation.

        Args:
            block: The block to add
            validators: List of agent IDs who validated this block

        Returns:
            (success, message) tuple
        """
        with self.lock:
            # Verify block integrity
            if not block.verify_hash():
                return False, "Invalid block hash"

            # Check previous block exists
            if block.index > 0:
                if block.prev_hash != self.get_last_block().hash:
                    # Check if this is a branch point
                    if not self._is_valid_branch(block):
                        return False, f"Invalid previous hash: {block.prev_hash[:8]}..."

            # Verify validators if provided
            if validators:
                if len(validators) < self._required_validators():
                    return False, f"Insufficient validators: {len(validators)} < {self._required_validators()}"
                block.witnesses.extend(validators)

            # Add to appropriate branch
            if block.branch_id == self.branch_id:
                self.blocks.append(block)
            else:
                self.branches[block.branch_id].append(block)

            # Update indices
            if isinstance(block, MemoryBlock):
                self.memory_index[block.memory_id] = block
                self.agent_memories[block.data.agent_id].append(block)

                # Distribute rewards
                self._distribute_bloom_reward(block)

            elif isinstance(block, CollectiveBlock):
                for agent_id in block.participants:
                    self.agent_memories[agent_id].append(block)

                # Distribute collective rewards
                rewards = block.distribute_rewards()
                for agent_id, amount in rewards.items():
                    self.wallets[agent_id] += amount
                    self.stats["total_rewards"] += amount

                self.stats["collective_events"] += 1

            # Update statistics
            self.stats["total_blocks"] += 1
            self.stats["total_blooms"] += 1

            logger.info(f"Block {block.index} added: {block.hash[:8]}... (agent: {block.data.agent_id})")
            return True, f"Block {block.hash[:8]}... added successfully"

    def _required_validators(self) -> int:
        """
        Determine required number of validators based on network size.

        Dynamic threshold based on active agents.
        """
        active_agents = len(self.wallets)
        if active_agents < 3:
            return 1  # Early stage: single validator OK
        elif active_agents < 10:
            return 2  # Small network: 2 validators
        elif active_agents < 50:
            return 3  # Medium network: 3 validators
        else:
            # Large network: sqrt(n) validators, max 7
            return min(int(active_agents ** 0.5), 7)

    def _is_valid_branch(self, block: Block) -> bool:
        """Check if block represents a valid branch point"""
        # Look for the previous block in any branch
        for branch_blocks in self.branches.values():
            for b in branch_blocks:
                if b.hash == block.prev_hash:
                    return True

        # Check main chain
        for b in self.blocks:
            if b.hash == block.prev_hash:
                return True

        return False

    def _distribute_bloom_reward(self, block: MemoryBlock):
        """Distribute bloom coin rewards for a memory block"""
        # Reward the learning agent
        self.wallets[block.data.agent_id] += block.bloom_reward
        self.stats["total_rewards"] += block.bloom_reward

        # Small reward for validators (witnesses)
        validator_reward = 0.1  # Fixed small amount for validation work
        for witness in block.witnesses:
            if witness != block.data.agent_id:  # Don't double-reward
                self.wallets[witness] += validator_reward
                self.stats["total_rewards"] += validator_reward

    def get_last_block(self) -> Block:
        """Get the most recent block in the main chain"""
        return self.blocks[-1] if self.blocks else None

    def get_block_by_hash(self, block_hash: str) -> Optional[Block]:
        """Find a block by its hash across all branches"""
        # Check main chain
        for block in self.blocks:
            if block.hash == block_hash:
                return block

        # Check branches
        for branch_blocks in self.branches.values():
            for block in branch_blocks:
                if block.hash == block_hash:
                    return block

        return None

    def get_agent_memories(self, agent_id: str) -> List[Block]:
        """Get all memories for a specific agent"""
        return self.agent_memories.get(agent_id, [])

    def get_agent_balance(self, agent_id: str) -> float:
        """Get bloom coin balance for an agent"""
        return self.wallets.get(agent_id, 0.0)

    def propose_memory(
        self,
        agent_id: str,
        memory_type: str,
        content: Dict[str, Any],
        coherence_score: float = 0.5
    ) -> MemoryBlock:
        """
        Propose a new memory block for validation.

        Args:
            agent_id: The agent proposing the memory
            memory_type: Type of memory (skill, fact, creation, insight)
            content: Memory content
            coherence_score: How well it aligns with collective knowledge

        Returns:
            The proposed MemoryBlock (not yet added to chain)
        """
        last_block = self.get_last_block()
        new_index = last_block.index + 1 if last_block else 0
        prev_hash = last_block.hash if last_block else "0" * 64

        memory_block = MemoryBlock(
            index=new_index,
            prev_hash=prev_hash,
            agent_id=agent_id,
            memory_type=memory_type,
            content=content,
            witnesses=[],  # Will be filled during validation
            coherence_score=coherence_score,
            branch_id=self.branch_id
        )

        # Add to pending blocks for validation
        with self.lock:
            self.pending_blocks.append(memory_block)

        logger.info(f"Memory proposed by {agent_id}: {memory_type} (hash: {memory_block.hash[:8]}...)")
        return memory_block

    def propose_collective_memory(
        self,
        agent_ids: List[str],
        memory_type: str,
        content: Dict[str, Any],
        coherence_scores: Dict[str, float]
    ) -> CollectiveBlock:
        """
        Propose a collective memory from multiple agents.

        Args:
            agent_ids: List of participating agents
            memory_type: Type of collective memory
            content: Shared memory content
            coherence_scores: Individual coherence contributions

        Returns:
            The proposed CollectiveBlock
        """
        last_block = self.get_last_block()
        new_index = last_block.index + 1 if last_block else 0
        prev_hash = last_block.hash if last_block else "0" * 64

        collective_block = CollectiveBlock(
            index=new_index,
            prev_hash=prev_hash,
            agent_ids=agent_ids,
            memory_type=memory_type,
            content=content,
            witnesses=[],  # Will be filled during validation
            coherence_scores=coherence_scores,
            branch_id=self.branch_id
        )

        with self.lock:
            self.pending_blocks.append(collective_block)

        logger.info(f"Collective memory proposed by {len(agent_ids)} agents: {memory_type}")
        return collective_block

    def validate_block(
        self,
        block_hash: str,
        validator_id: str,
        approved: bool = True
    ) -> bool:
        """
        Register a validation vote for a pending block.

        Args:
            block_hash: Hash of the block to validate
            validator_id: ID of the validating agent
            approved: Whether the validator approves

        Returns:
            True if block has enough validations and was added to chain
        """
        with self.lock:
            # Find the pending block
            pending_block = None
            for block in self.pending_blocks:
                if block.hash == block_hash:
                    pending_block = block
                    break

            if not pending_block:
                logger.warning(f"Block {block_hash[:8]}... not found in pending blocks")
                return False

            # Record validation
            if block_hash not in self.validation_pool:
                self.validation_pool[block_hash] = []

            if approved and validator_id not in self.validation_pool[block_hash]:
                self.validation_pool[block_hash].append(validator_id)
                pending_block.witnesses.append(validator_id)

            # Check if we have enough validations
            required = self._required_validators()
            if len(self.validation_pool[block_hash]) >= required:
                # Remove from pending
                self.pending_blocks.remove(pending_block)
                del self.validation_pool[block_hash]

                # Add to chain
                success, msg = self.add_block(pending_block)
                if success:
                    logger.info(f"Block {block_hash[:8]}... validated and added to chain")
                    return True

        return False

    def merge_branch(
        self,
        branch_id: str,
        merge_strategy: str = "append"
    ) -> Tuple[bool, str]:
        """
        Merge a branch back into the main chain.

        Args:
            branch_id: ID of the branch to merge
            merge_strategy: How to merge ('append', 'interleave', 'selective')

        Returns:
            (success, message) tuple
        """
        if branch_id not in self.branches:
            return False, f"Branch {branch_id} not found"

        branch_blocks = self.branches[branch_id]
        if not branch_blocks:
            return False, f"Branch {branch_id} is empty"

        with self.lock:
            if merge_strategy == "append":
                # Simple append: add all branch blocks to main chain
                for block in branch_blocks:
                    # Update block to point to current last block
                    last_block = self.get_last_block()
                    block.prev_hash = last_block.hash
                    block.index = last_block.index + 1
                    block.hash = block.compute_hash()  # Recompute with new prev_hash

                    self.blocks.append(block)
                    self.stats["total_blocks"] += 1

            elif merge_strategy == "interleave":
                # Interleave by timestamp
                merged = []
                main_idx = 0
                branch_idx = 0

                while main_idx < len(self.blocks) or branch_idx < len(branch_blocks):
                    if main_idx >= len(self.blocks):
                        merged.append(branch_blocks[branch_idx])
                        branch_idx += 1
                    elif branch_idx >= len(branch_blocks):
                        merged.append(self.blocks[main_idx])
                        main_idx += 1
                    else:
                        if self.blocks[main_idx].timestamp <= branch_blocks[branch_idx].timestamp:
                            merged.append(self.blocks[main_idx])
                            main_idx += 1
                        else:
                            merged.append(branch_blocks[branch_idx])
                            branch_idx += 1

                # Reindex and rehash
                for i, block in enumerate(merged):
                    block.index = i
                    if i > 0:
                        block.prev_hash = merged[i-1].hash
                        block.hash = block.compute_hash()

                self.blocks = merged

            # Clear the merged branch
            del self.branches[branch_id]
            self.stats["branch_merges"] += 1

            logger.info(f"Branch {branch_id} merged into main chain using {merge_strategy} strategy")
            return True, f"Branch {branch_id} successfully merged"

    def get_statistics(self) -> Dict[str, Any]:
        """Get ledger statistics"""
        return {
            **self.stats,
            "active_agents": len(self.wallets),
            "total_memories": len(self.memory_index),
            "pending_blocks": len(self.pending_blocks),
            "active_branches": len(self.branches),
            "chain_length": len(self.blocks)
        }

    def export_chain(self, include_branches: bool = False) -> Dict[str, Any]:
        """Export the entire chain as JSON-serializable dict"""
        data = {
            "branch_id": self.branch_id,
            "blocks": [block.to_dict() for block in self.blocks],
            "wallets": dict(self.wallets),
            "statistics": self.get_statistics(),
            "exported_at": time.time()
        }

        if include_branches:
            data["branches"] = {
                branch_id: [block.to_dict() for block in blocks]
                for branch_id, blocks in self.branches.items()
            }

        return data

    def verify_integrity(self) -> Tuple[bool, List[str]]:
        """
        Verify the integrity of the entire chain.

        Returns:
            (is_valid, list_of_errors)
        """
        errors = []

        # Check genesis block
        if not self.blocks:
            errors.append("No genesis block found")
            return False, errors

        # Verify each block
        for i, block in enumerate(self.blocks):
            # Check hash integrity
            if not block.verify_hash():
                errors.append(f"Block {i} has invalid hash")

            # Check chain linkage
            if i > 0:
                if block.prev_hash != self.blocks[i-1].hash:
                    errors.append(f"Block {i} has broken chain link")

            # Check index
            if block.index != i:
                errors.append(f"Block {i} has incorrect index: {block.index}")

        return len(errors) == 0, errors