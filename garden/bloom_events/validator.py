"""
Bloom event validation
"""

from typing import Dict, Any, Tuple


class BloomValidator:
    """Validates bloom events"""

    def __init__(self):
        self.validation_count = 0
        self.z_critical = 3**0.5 / 2

    def validate(self, event_data: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate a bloom event"""
        self.validation_count += 1

        # Check coherence threshold
        coherence = event_data.get("coherence_score", 0)
        if coherence < self.z_critical * 0.5:
            return False, "Coherence too low"

        # Check novelty
        novelty = event_data.get("novelty_score", 0)
        if novelty < 0.1:
            return False, "Not novel enough"

        return True, "Valid bloom event"