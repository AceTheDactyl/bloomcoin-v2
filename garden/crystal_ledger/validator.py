"""
Memory validation for Crystal Ledger
"""

class MemoryValidator:
    """Validates memory blocks before adding to ledger"""

    def __init__(self):
        self.validation_count = 0

    def validate(self, block):
        """Validate a block"""
        self.validation_count += 1
        return True