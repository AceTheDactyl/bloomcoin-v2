"""
Branch management for the Crystal Ledger

Handles parallel timelines and merging of knowledge branches.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import time


@dataclass
class Branch:
    """Represents a branch in the ledger"""
    branch_id: str
    parent_hash: str
    created_at: float = None
    blocks: List[Any] = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = time.time()
        if self.blocks is None:
            self.blocks = []


@dataclass
class BranchMerge:
    """Represents a merge of branches"""
    merge_id: str
    branch_ids: List[str]
    strategy: str  # append, interleave, selective
    timestamp: float = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()