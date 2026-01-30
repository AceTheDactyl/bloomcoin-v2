"""
Learning engine for AI agents
"""

from typing import Dict, Any, Optional


class LearningEngine:
    """Manages agent learning processes"""

    def __init__(self):
        self.learning_history = []

    def process_information(self, information: Dict[str, Any]) -> bool:
        """Process new information for learning"""
        self.learning_history.append(information)
        return True


class BloomDetector:
    """Detects when learning qualifies as a Bloom event"""

    def __init__(self):
        self.detection_threshold = 0.7

    def is_bloom_worthy(self, learning_data: Dict[str, Any]) -> bool:
        """Check if learning qualifies as a Bloom event"""
        # Simplified detection logic
        coherence = learning_data.get("coherence", 0)
        novelty = learning_data.get("novelty", 0)

        score = (coherence + novelty) / 2
        return score >= self.detection_threshold