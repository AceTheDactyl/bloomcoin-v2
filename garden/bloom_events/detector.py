"""
Bloom event detection system
"""

from typing import Dict, Any
from .bloom_event import BloomType


class BloomDetector:
    """Detects and classifies bloom events"""

    def __init__(self):
        self.detection_threshold = 0.7
        self.z_critical = 3**0.5 / 2

    def detect(self, agent_data: Dict[str, Any]) -> BloomType:
        """Detect the type of bloom event"""
        # Classify based on content
        if "skill" in agent_data.get("type", ""):
            return BloomType.SKILL_ACQUISITION
        elif "creation" in agent_data.get("type", ""):
            return BloomType.CREATION
        elif "insight" in agent_data.get("type", ""):
            return BloomType.INSIGHT
        elif "collaboration" in agent_data.get("type", ""):
            return BloomType.COLLABORATION
        else:
            return BloomType.LEARNING

    def calculate_significance(self, event_data: Dict[str, Any]) -> float:
        """Calculate the significance of an event"""
        coherence = event_data.get("coherence", 0.5)
        novelty = event_data.get("novelty", 0.5)

        # Use critical coherence threshold
        coherence_factor = 1.0 - abs(coherence - self.z_critical)
        return (coherence_factor + novelty) / 2