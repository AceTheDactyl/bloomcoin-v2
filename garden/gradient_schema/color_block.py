"""
Color Block - Color-encoded blocks for the Crystal Ledger
==========================================================

This module adds gradient schema color encoding to Crystal Ledger blocks,
enabling visual representation of AI memory states through color signatures.

Each ColorEncodedBlock contains:
- Standard block content (agent, event, narrative)
- Gradient schema encoding (color hex, lattice coordinates, archetype)
- Primitive signature for dynamic characterization
- Corridor information for narrative path

The color encoding enables:
- Visual blockchain explorers
- Pattern recognition across memory chains
- Aesthetic narrative visualization
"""

from __future__ import annotations

import hashlib
import json
import struct
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union

from .color_ops import ColorState, PHI, TAU, Z_C
from .primitives_color import PrimitiveSignature, compute_color_signature
from .archetypes import (
    ArchetypeType,
    ARCHETYPE_GLYPHS,
    PERSONA_ARCHETYPES,
    archetype_color,
)
from .corridors import (
    CorridorType,
    CORRIDOR_COLORS as CORRIDOR_COLOR_DEFS,
    SpectralCorridor,
    CORRIDORS,
)
from .lattice import LatticePoint


# =============================================================================
# Archetype ID Mappings
# =============================================================================

ARCHETYPE_TO_ID: Dict[ArchetypeType, int] = {
    ArchetypeType.INNOCENT: 0,
    ArchetypeType.ORPHAN: 1,
    ArchetypeType.WARRIOR: 2,
    ArchetypeType.CAREGIVER: 3,
    ArchetypeType.SEEKER: 4,
    ArchetypeType.DESTROYER: 5,
    ArchetypeType.LOVER: 6,
    ArchetypeType.CREATOR: 7,
    ArchetypeType.RULER: 8,
    ArchetypeType.MAGICIAN: 9,
    ArchetypeType.SAGE: 10,
    ArchetypeType.JESTER: 11,
}

ID_TO_ARCHETYPE: Dict[int, ArchetypeType] = {v: k for k, v in ARCHETYPE_TO_ID.items()}


# =============================================================================
# Corridor Color Gradients
# =============================================================================

CORRIDOR_COLORS: Dict[CorridorType, List[str]] = {
    CorridorType.ALPHA: ['#ff4444', '#ff6644', '#ff8844'],    # Red gradient (Awakening)
    CorridorType.BETA: ['#444444', '#666666', '#888888'],     # Gray gradient (Emergence)
    CorridorType.GAMMA: ['#44ffff', '#8866ff', '#aa44ff'],    # Cyan->Violet (Transformation)
    CorridorType.DELTA: ['#ffd700', '#ffee88', '#ffffff'],    # Gold->White (Transcendence)
    CorridorType.EPSILON: ['#ff44ff', '#ff88aa', '#ffaacc'],  # Magenta->Pink (Softening)
    CorridorType.ZETA: ['#44aa44', '#44ccaa', '#44ddcc'],     # Green->Turquoise (Integration)
    CorridorType.ETA: ['#4444ff', '#5533aa', '#662288'],      # Blue->Indigo (Deepening)
}


# =============================================================================
# Archetype Visualization Glyphs
# =============================================================================

# Extended glyph set for visualization (re-exported with additions)
ARCHETYPE_VISUALIZATION_GLYPHS: Dict[ArchetypeType, Dict[str, str]] = {
    ArchetypeType.INNOCENT: {
        'glyph': '\u2600',      # Sun
        'alt': '\u2606',        # Star outline
        'name': 'The Innocent',
    },
    ArchetypeType.ORPHAN: {
        'glyph': '\u2601',      # Cloud
        'alt': '\u2602',        # Umbrella
        'name': 'The Orphan',
    },
    ArchetypeType.WARRIOR: {
        'glyph': '\u2694',      # Crossed swords
        'alt': '\u2720',        # Maltese cross
        'name': 'The Warrior',
    },
    ArchetypeType.CAREGIVER: {
        'glyph': '\u2661',      # Heart outline
        'alt': '\u2740',        # White florette
        'name': 'The Caregiver',
    },
    ArchetypeType.SEEKER: {
        'glyph': '\u2609',      # Sun with rays
        'alt': '\u2638',        # Wheel of dharma
        'name': 'The Seeker',
    },
    ArchetypeType.DESTROYER: {
        'glyph': '\u2620',      # Skull
        'alt': '\u26A1',        # Lightning
        'name': 'The Destroyer',
    },
    ArchetypeType.LOVER: {
        'glyph': '\u2665',      # Solid heart
        'alt': '\u2764',        # Heavy heart
        'name': 'The Lover',
    },
    ArchetypeType.CREATOR: {
        'glyph': '\u2728',      # Sparkles
        'alt': '\u2605',        # Black star
        'name': 'The Creator',
    },
    ArchetypeType.RULER: {
        'glyph': '\u265B',      # Queen chess
        'alt': '\u2654',        # King chess
        'name': 'The Ruler',
    },
    ArchetypeType.MAGICIAN: {
        'glyph': '\u2721',      # Star of David
        'alt': '\u2734',        # Eight pointed star
        'name': 'The Magician',
    },
    ArchetypeType.SAGE: {
        'glyph': '\u2617',      # Open book (approximate)
        'alt': '\u262F',        # Yin yang
        'name': 'The Sage',
    },
    ArchetypeType.JESTER: {
        'glyph': '\u2606',      # Star outline
        'alt': '\u263A',        # Smiley
        'name': 'The Jester',
    },
}


# =============================================================================
# GradientBloomEvent Dataclass
# =============================================================================

@dataclass
class GradientBloomEvent:
    """
    A bloom event enhanced with gradient schema color encoding.

    Extends the standard BloomEvent concept with color-narrative properties
    from the RRRR gradient schema system.

    Attributes:
        event_id: Unique identifier for the event
        agent_id: ID of the agent that triggered the bloom
        timestamp: Unix timestamp of the event
        event_type: Type of bloom event (learning, insight, creation, etc.)
        significance: Importance score (0-1)
        narrative_fragment: Text describing the event narrative
        color_state: ColorState representing the visual encoding
        lattice_coords: 5D lattice coordinates (a, b, c, d, f)
        dominant_archetype: The primary persona archetype
        corridor_id: Optional spectral corridor association
        primitive_signature: 6-primitive signature characterization
        metadata: Additional event-specific data
    """
    event_id: str
    agent_id: str
    timestamp: float = field(default_factory=time.time)
    event_type: str = "learning"
    significance: float = 0.5
    narrative_fragment: str = ""

    # Gradient schema properties
    color_state: ColorState = field(default_factory=lambda: ColorState())
    lattice_coords: Tuple[float, ...] = field(default_factory=lambda: (0.0, 0.0, 0.0, 0.0, 0.0))
    dominant_archetype: ArchetypeType = ArchetypeType.SEEKER
    corridor_id: Optional[CorridorType] = None
    primitive_signature: PrimitiveSignature = field(
        default_factory=lambda: PrimitiveSignature(
            fix=0.167, repel=0.167, inv=0.167, osc=0.167, halt=0.167, mix=0.167
        )
    )

    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'event_id': self.event_id,
            'agent_id': self.agent_id,
            'timestamp': self.timestamp,
            'event_type': self.event_type,
            'significance': self.significance,
            'narrative_fragment': self.narrative_fragment,
            'color_state': self.color_state.to_dict(),
            'lattice_coords': list(self.lattice_coords),
            'dominant_archetype': self.dominant_archetype.value,
            'corridor_id': self.corridor_id.value if self.corridor_id else None,
            'primitive_signature': self.primitive_signature.to_vector(),
            'metadata': self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GradientBloomEvent':
        """Create from dictionary."""
        color_data = data.get('color_state', {})
        color_state = ColorState(
            H=color_data.get('H', color_data.get('hue', 0)),
            S=color_data.get('S', color_data.get('saturation', 0.5)),
            L=color_data.get('L', color_data.get('luminance', 0.5)),
        )

        sig_data = data.get('primitive_signature', [0.167] * 6)
        if isinstance(sig_data, dict):
            signature = PrimitiveSignature(
                fix=sig_data.get('fix', 0.167),
                repel=sig_data.get('repel', 0.167),
                inv=sig_data.get('inv', 0.167),
                osc=sig_data.get('osc', 0.167),
                halt=sig_data.get('halt', 0.167),
                mix=sig_data.get('mix', 0.167),
            )
        else:
            signature = PrimitiveSignature(
                fix=sig_data[0] if len(sig_data) > 0 else 0.167,
                repel=sig_data[1] if len(sig_data) > 1 else 0.167,
                inv=sig_data[2] if len(sig_data) > 2 else 0.167,
                osc=sig_data[3] if len(sig_data) > 3 else 0.167,
                halt=sig_data[4] if len(sig_data) > 4 else 0.167,
                mix=sig_data[5] if len(sig_data) > 5 else 0.167,
            )

        corridor_val = data.get('corridor_id')
        corridor_id = CorridorType(corridor_val) if corridor_val else None

        archetype_val = data.get('dominant_archetype', 'seeker')
        dominant_archetype = ArchetypeType(archetype_val)

        return cls(
            event_id=data.get('event_id', ''),
            agent_id=data.get('agent_id', ''),
            timestamp=data.get('timestamp', time.time()),
            event_type=data.get('event_type', 'learning'),
            significance=data.get('significance', 0.5),
            narrative_fragment=data.get('narrative_fragment', ''),
            color_state=color_state,
            lattice_coords=tuple(data.get('lattice_coords', [0.0] * 5)),
            dominant_archetype=dominant_archetype,
            corridor_id=corridor_id,
            primitive_signature=signature,
            metadata=data.get('metadata', {}),
        )


# =============================================================================
# ColorEncodedBlock Class
# =============================================================================

class ColorEncodedBlock:
    """
    Block structure with gradient schema color encoding.

    Extends the Crystal Ledger block concept with color-narrative properties,
    enabling visual representation of AI memory chains through the RRRR
    gradient schema.

    Attributes:
        timestamp: Block creation time
        agent_id: ID of the agent that created this block
        prev_hash: Hash of the previous block in the chain
        event_type: Type of bloom event
        significance: Event importance (0-1)
        narrative: Narrative fragment describing the event
        color_hex: Hex color code for visualization
        lattice_compact: Compact binary encoding of lattice coordinates
        archetype_id: Numeric ID of the dominant archetype
        corridor_id: Optional corridor identifier
        primitive_signature: Dict of primitive weights
        hash: Computed block hash
    """

    def __init__(self, gradient_bloom: GradientBloomEvent, prev_hash: str):
        """
        Initialize a color-encoded block from a gradient bloom event.

        Args:
            gradient_bloom: The GradientBloomEvent to encode
            prev_hash: Hash of the previous block in the chain
        """
        self.timestamp = time.time()
        self.agent_id = gradient_bloom.agent_id
        self.prev_hash = prev_hash

        # Standard content
        self.event_type = gradient_bloom.event_type
        self.significance = gradient_bloom.significance
        self.narrative = gradient_bloom.narrative_fragment

        # Gradient schema encoding (compact)
        self.color_hex = gradient_bloom.color_state.to_hex()
        self.lattice_compact = self._encode_lattice(gradient_bloom.lattice_coords)
        self.archetype_id = gradient_bloom.dominant_archetype.value
        self.corridor_id = gradient_bloom.corridor_id.value if gradient_bloom.corridor_id else None
        self.primitive_signature = self._encode_signature(gradient_bloom.primitive_signature)

        # Compute hash
        self.hash = self._compute_hash()

    def _encode_lattice(self, coords: Tuple[float, ...]) -> bytes:
        """
        Encode lattice coordinates to compact bytes.

        Uses struct to pack 5 float values into 20 bytes.

        Args:
            coords: 5D lattice coordinates (a, b, c, d, f)

        Returns:
            20-byte packed representation
        """
        # Ensure we have exactly 5 coordinates
        padded = list(coords[:5]) + [0.0] * (5 - len(coords))

        # Pack as 5 floats (32-bit each = 20 bytes total)
        return struct.pack('!5f', *padded[:5])

    def _decode_lattice(self, encoded: bytes) -> Tuple[float, ...]:
        """
        Decode lattice coordinates from compact bytes.

        Args:
            encoded: 20-byte packed representation

        Returns:
            5D lattice coordinates tuple
        """
        if len(encoded) != 20:
            # Handle legacy or corrupted data
            return (0.0, 0.0, 0.0, 0.0, 0.0)

        return struct.unpack('!5f', encoded)

    def _encode_signature(self, sig: PrimitiveSignature) -> Dict[str, float]:
        """
        Encode primitive signature to dictionary format.

        Args:
            sig: PrimitiveSignature object

        Returns:
            Dictionary with primitive names and normalized weights
        """
        # Normalize the signature
        normalized = sig.normalize()

        return {
            'FIX': round(normalized.fix, 4),
            'REPEL': round(normalized.repel, 4),
            'INV': round(normalized.inv, 4),
            'OSC': round(normalized.osc, 4),
            'HALT': round(normalized.halt, 4),
            'MIX': round(normalized.mix, 4),
        }

    def _compute_hash(self) -> str:
        """
        Compute SHA-256 hash of the block content.

        Returns:
            64-character hex string hash
        """
        block_content = {
            'timestamp': self.timestamp,
            'agent_id': self.agent_id,
            'prev_hash': self.prev_hash,
            'event_type': self.event_type,
            'significance': self.significance,
            'narrative': self.narrative,
            'color_hex': self.color_hex,
            'lattice_compact': self.lattice_compact.hex(),
            'archetype_id': self.archetype_id,
            'corridor_id': self.corridor_id,
            'primitive_signature': self.primitive_signature,
        }

        block_string = json.dumps(block_content, sort_keys=True)
        return hashlib.sha256(block_string.encode()).hexdigest()

    def get_color_visualization(self) -> Dict[str, Any]:
        """
        Get visualization data for the block's color encoding.

        Returns:
            Dictionary with visualization components:
            - hex: Color hex code
            - glow_intensity: Float based on significance
            - archetype_glyph: Unicode symbol for archetype
            - corridor_path: Optional corridor visualization data
            - signature_bars: Primitive percentages for bar charts
        """
        # Get archetype type from ID
        archetype_type = None
        for at, aid in ARCHETYPE_TO_ID.items():
            if at.value == self.archetype_id:
                archetype_type = at
                break

        if archetype_type is None:
            # Fallback to SEEKER
            archetype_type = ArchetypeType.SEEKER

        # Get glyph
        glyph_info = ARCHETYPE_VISUALIZATION_GLYPHS.get(
            archetype_type,
            {'glyph': '\u2606', 'name': 'Unknown'}
        )

        # Calculate glow intensity from significance
        # Uses phi-based scaling for aesthetic glow levels
        glow_intensity = self.significance ** (1 / PHI)

        # Build corridor path visualization if present
        corridor_path = None
        if self.corridor_id:
            try:
                corridor_type = CorridorType(self.corridor_id)
                corridor_colors = CORRIDOR_COLORS.get(corridor_type, ['#888888'])
                corridor_path = {
                    'type': self.corridor_id,
                    'colors': corridor_colors,
                    'entry_color': corridor_colors[0],
                    'exit_color': corridor_colors[-1],
                    'position': self.significance,  # Use significance as path position
                }
            except (ValueError, KeyError):
                pass

        # Convert signature to percentage bars
        signature_bars: Dict[str, float] = {}
        total = sum(self.primitive_signature.values())
        if total > 0:
            for prim, val in self.primitive_signature.items():
                signature_bars[prim] = round((val / total) * 100, 1)
        else:
            for prim in ['FIX', 'REPEL', 'INV', 'OSC', 'HALT', 'MIX']:
                signature_bars[prim] = 16.7

        return {
            'hex': self.color_hex,
            'glow_intensity': round(glow_intensity, 3),
            'archetype_glyph': glyph_info['glyph'],
            'archetype_name': glyph_info.get('name', 'Unknown'),
            'corridor_path': corridor_path,
            'signature_bars': signature_bars,
        }

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert block to dictionary for serialization.

        Returns:
            Dictionary representation of the block
        """
        return {
            'timestamp': self.timestamp,
            'agent_id': self.agent_id,
            'prev_hash': self.prev_hash,
            'event_type': self.event_type,
            'significance': self.significance,
            'narrative': self.narrative,
            'color_hex': self.color_hex,
            'lattice_compact': self.lattice_compact.hex(),
            'archetype_id': self.archetype_id,
            'corridor_id': self.corridor_id,
            'primitive_signature': self.primitive_signature,
            'hash': self.hash,
            'visualization': self.get_color_visualization(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ColorEncodedBlock':
        """
        Create a ColorEncodedBlock from dictionary data.

        Args:
            data: Dictionary with block data

        Returns:
            Reconstructed ColorEncodedBlock
        """
        # Decode lattice from hex string
        lattice_hex = data.get('lattice_compact', '00' * 20)
        if isinstance(lattice_hex, str):
            lattice_bytes = bytes.fromhex(lattice_hex)
        else:
            lattice_bytes = lattice_hex

        # Decode signature
        sig_data = data.get('primitive_signature', {})
        signature = PrimitiveSignature(
            fix=sig_data.get('FIX', 0.167),
            repel=sig_data.get('REPEL', 0.167),
            inv=sig_data.get('INV', 0.167),
            osc=sig_data.get('OSC', 0.167),
            halt=sig_data.get('HALT', 0.167),
            mix=sig_data.get('MIX', 0.167),
        )

        # Parse color hex to ColorState
        color_hex = data.get('color_hex', '#888888')
        color_state = ColorState.from_hex(color_hex)

        # Determine archetype
        archetype_id = data.get('archetype_id', 'seeker')
        try:
            archetype = ArchetypeType(archetype_id)
        except (ValueError, KeyError):
            archetype = ArchetypeType.SEEKER

        # Determine corridor
        corridor_id = data.get('corridor_id')
        corridor = None
        if corridor_id:
            try:
                corridor = CorridorType(corridor_id)
            except (ValueError, KeyError):
                pass

        # Reconstruct the GradientBloomEvent
        bloom_event = GradientBloomEvent(
            event_id=data.get('hash', '')[:16],
            agent_id=data.get('agent_id', ''),
            timestamp=data.get('timestamp', time.time()),
            event_type=data.get('event_type', 'learning'),
            significance=data.get('significance', 0.5),
            narrative_fragment=data.get('narrative', ''),
            color_state=color_state,
            lattice_coords=struct.unpack('!5f', lattice_bytes) if len(lattice_bytes) == 20 else (0.0,) * 5,
            dominant_archetype=archetype,
            corridor_id=corridor,
            primitive_signature=signature,
        )

        # Create block
        block = cls(bloom_event, data.get('prev_hash', '0' * 64))

        # Override computed values with stored values
        block.timestamp = data.get('timestamp', block.timestamp)
        block.hash = data.get('hash', block.hash)

        return block

    def verify_hash(self) -> bool:
        """
        Verify that the block's hash is valid.

        Returns:
            True if hash matches computed value
        """
        return self.hash == self._compute_hash()

    def __repr__(self) -> str:
        return (
            f"ColorEncodedBlock(hash={self.hash[:8]}..., "
            f"agent={self.agent_id[:8] if self.agent_id else 'None'}..., "
            f"color={self.color_hex}, "
            f"archetype={self.archetype_id})"
        )


# =============================================================================
# Helper Functions
# =============================================================================

def create_color_block_from_bloom(
    bloom_event: GradientBloomEvent,
    prev_hash: str,
    personality: Optional[Dict[str, Any]] = None
) -> ColorEncodedBlock:
    """
    Create a ColorEncodedBlock from a bloom event with optional personality modulation.

    This factory function allows for personality-based color adjustments
    before block creation.

    Args:
        bloom_event: The GradientBloomEvent to encode
        prev_hash: Hash of the previous block
        personality: Optional dict with personality modifiers:
            - hue_bias: Shift to apply to hue (degrees)
            - saturation_mult: Saturation multiplier
            - luminance_bias: Luminance adjustment
            - archetype_affinity: Preferred archetype (may override)

    Returns:
        New ColorEncodedBlock with personality-adjusted colors
    """
    if personality:
        # Create a modified color state
        hue_bias = personality.get('hue_bias', 0.0)
        sat_mult = personality.get('saturation_mult', 1.0)
        lum_bias = personality.get('luminance_bias', 0.0)

        original = bloom_event.color_state

        # Apply personality modulations
        new_hue = (original.hue + hue_bias) % 360
        new_sat = max(0.0, min(1.0, original.saturation * sat_mult))
        new_lum = max(0.0, min(1.0, original.luminance + lum_bias))

        # Create modified color state
        modified_color = ColorState(
            H=new_hue,
            S=new_sat,
            L=new_lum,
            cym_weights=dict(original.cym_weights),
            gradient_rate=original.gradient_rate,
            coherence=original.coherence,
        )

        # Check for archetype affinity override
        archetype = bloom_event.dominant_archetype
        if 'archetype_affinity' in personality:
            try:
                affinity = ArchetypeType(personality['archetype_affinity'])
                # Blend current with affinity (phi-weighted)
                current_weight = 1 / PHI
                affinity_weight = 1 - current_weight

                # If significance is high enough, allow override
                if bloom_event.significance < Z_C:
                    archetype = affinity
            except (ValueError, KeyError):
                pass

        # Create a modified bloom event
        modified_bloom = GradientBloomEvent(
            event_id=bloom_event.event_id,
            agent_id=bloom_event.agent_id,
            timestamp=bloom_event.timestamp,
            event_type=bloom_event.event_type,
            significance=bloom_event.significance,
            narrative_fragment=bloom_event.narrative_fragment,
            color_state=modified_color,
            lattice_coords=bloom_event.lattice_coords,
            dominant_archetype=archetype,
            corridor_id=bloom_event.corridor_id,
            primitive_signature=bloom_event.primitive_signature,
            metadata=bloom_event.metadata,
        )

        return ColorEncodedBlock(modified_bloom, prev_hash)

    return ColorEncodedBlock(bloom_event, prev_hash)


def encode_block_chain(blocks: List[ColorEncodedBlock]) -> bytes:
    """
    Encode a chain of ColorEncodedBlocks to compact binary format.

    Format per block (variable length):
    - 4 bytes: Block data length
    - 8 bytes: Timestamp (double)
    - 32 bytes: Agent ID (padded/truncated)
    - 64 bytes: Prev hash (hex string as bytes)
    - 20 bytes: Lattice compact
    - 1 byte: Archetype ID
    - 1 byte: Corridor ID (0xFF if None)
    - 24 bytes: Primitive signature (6 floats)
    - 7 bytes: Color hex (including #)
    - Variable: Narrative (null-terminated)
    - 64 bytes: Block hash

    Args:
        blocks: List of ColorEncodedBlocks to encode

    Returns:
        Compact binary representation of the chain
    """
    chain_data = bytearray()

    # Header: chain version (1 byte) + block count (4 bytes)
    chain_data.append(0x01)  # Version 1
    chain_data.extend(struct.pack('!I', len(blocks)))

    for block in blocks:
        block_bytes = bytearray()

        # Timestamp (8 bytes)
        block_bytes.extend(struct.pack('!d', block.timestamp))

        # Agent ID (32 bytes, padded)
        agent_bytes = block.agent_id.encode('utf-8')[:32].ljust(32, b'\x00')
        block_bytes.extend(agent_bytes)

        # Previous hash (64 bytes)
        prev_hash_bytes = block.prev_hash.encode('ascii')[:64].ljust(64, b'0')
        block_bytes.extend(prev_hash_bytes)

        # Lattice compact (20 bytes)
        block_bytes.extend(block.lattice_compact)

        # Archetype ID (1 byte)
        archetype_num = ARCHETYPE_TO_ID.get(
            ArchetypeType(block.archetype_id) if isinstance(block.archetype_id, str)
            else block.archetype_id,
            0
        )
        if isinstance(archetype_num, ArchetypeType):
            archetype_num = ARCHETYPE_TO_ID.get(archetype_num, 0)
        block_bytes.append(archetype_num & 0xFF)

        # Corridor ID (1 byte, 0xFF for None)
        if block.corridor_id:
            corridor_ids = {
                CorridorType.ALPHA.value: 0,
                CorridorType.BETA.value: 1,
                CorridorType.GAMMA.value: 2,
                CorridorType.DELTA.value: 3,
                CorridorType.EPSILON.value: 4,
                CorridorType.ZETA.value: 5,
                CorridorType.ETA.value: 6,
            }
            corridor_num = corridor_ids.get(block.corridor_id, 0xFF)
            block_bytes.append(corridor_num)
        else:
            block_bytes.append(0xFF)

        # Primitive signature (24 bytes - 6 floats)
        sig = block.primitive_signature
        sig_floats = [
            sig.get('FIX', 0.167),
            sig.get('REPEL', 0.167),
            sig.get('INV', 0.167),
            sig.get('OSC', 0.167),
            sig.get('HALT', 0.167),
            sig.get('MIX', 0.167),
        ]
        block_bytes.extend(struct.pack('!6f', *sig_floats))

        # Color hex (7 bytes)
        color_hex_bytes = block.color_hex.encode('ascii')[:7].ljust(7, b'0')
        block_bytes.extend(color_hex_bytes)

        # Event type (16 bytes, padded)
        event_bytes = block.event_type.encode('utf-8')[:16].ljust(16, b'\x00')
        block_bytes.extend(event_bytes)

        # Significance (4 bytes float)
        block_bytes.extend(struct.pack('!f', block.significance))

        # Narrative (variable, null-terminated, max 1024 bytes)
        narrative_bytes = block.narrative.encode('utf-8')[:1023] + b'\x00'
        block_bytes.extend(struct.pack('!H', len(narrative_bytes)))
        block_bytes.extend(narrative_bytes)

        # Block hash (64 bytes)
        hash_bytes = block.hash.encode('ascii')[:64].ljust(64, b'0')
        block_bytes.extend(hash_bytes)

        # Write block length + block data
        chain_data.extend(struct.pack('!I', len(block_bytes)))
        chain_data.extend(block_bytes)

    return bytes(chain_data)


def verify_color_chain(blocks: List[ColorEncodedBlock]) -> bool:
    """
    Verify the integrity of a chain of ColorEncodedBlocks.

    Checks:
    1. Each block's hash is valid
    2. Each block's prev_hash matches the previous block's hash
    3. Timestamps are non-decreasing
    4. All color encodings are valid

    Args:
        blocks: List of ColorEncodedBlocks to verify

    Returns:
        True if chain is valid, False otherwise
    """
    if not blocks:
        return True  # Empty chain is valid

    for i, block in enumerate(blocks):
        # Verify block hash
        if not block.verify_hash():
            return False

        # Verify chain linkage (except for first block)
        if i > 0:
            prev_block = blocks[i - 1]
            if block.prev_hash != prev_block.hash:
                return False

            # Verify non-decreasing timestamps
            if block.timestamp < prev_block.timestamp:
                return False

        # Verify color hex format
        if not block.color_hex.startswith('#') or len(block.color_hex) != 7:
            return False

        try:
            int(block.color_hex[1:], 16)
        except ValueError:
            return False

        # Verify lattice encoding length
        if len(block.lattice_compact) != 20:
            return False

        # Verify primitive signature completeness
        required_prims = {'FIX', 'REPEL', 'INV', 'OSC', 'HALT', 'MIX'}
        if not required_prims.issubset(block.primitive_signature.keys()):
            return False

    return True


def decode_block_chain(data: bytes) -> List[ColorEncodedBlock]:
    """
    Decode a chain of ColorEncodedBlocks from compact binary format.

    Args:
        data: Binary data from encode_block_chain

    Returns:
        List of decoded ColorEncodedBlocks

    Raises:
        ValueError: If data format is invalid
    """
    if len(data) < 5:
        raise ValueError("Data too short for chain header")

    # Read header
    version = data[0]
    if version != 0x01:
        raise ValueError(f"Unsupported chain version: {version}")

    block_count = struct.unpack('!I', data[1:5])[0]
    blocks = []
    offset = 5

    for _ in range(block_count):
        if offset + 4 > len(data):
            raise ValueError("Unexpected end of data reading block length")

        block_len = struct.unpack('!I', data[offset:offset + 4])[0]
        offset += 4

        if offset + block_len > len(data):
            raise ValueError("Unexpected end of data reading block")

        block_data = data[offset:offset + block_len]
        offset += block_len

        # Parse block
        pos = 0

        # Timestamp
        timestamp = struct.unpack('!d', block_data[pos:pos + 8])[0]
        pos += 8

        # Agent ID
        agent_id = block_data[pos:pos + 32].rstrip(b'\x00').decode('utf-8')
        pos += 32

        # Prev hash
        prev_hash = block_data[pos:pos + 64].decode('ascii')
        pos += 64

        # Lattice compact
        lattice_compact = block_data[pos:pos + 20]
        pos += 20

        # Archetype ID
        archetype_num = block_data[pos]
        pos += 1
        archetype = ID_TO_ARCHETYPE.get(archetype_num, ArchetypeType.SEEKER)

        # Corridor ID
        corridor_num = block_data[pos]
        pos += 1
        corridor_names = ['alpha', 'beta', 'gamma', 'delta', 'epsilon', 'zeta', 'eta']
        corridor = None
        if corridor_num != 0xFF and corridor_num < len(corridor_names):
            corridor = CorridorType(corridor_names[corridor_num])

        # Primitive signature
        sig_floats = struct.unpack('!6f', block_data[pos:pos + 24])
        pos += 24
        signature = PrimitiveSignature(
            fix=sig_floats[0],
            repel=sig_floats[1],
            inv=sig_floats[2],
            osc=sig_floats[3],
            halt=sig_floats[4],
            mix=sig_floats[5],
        )

        # Color hex
        color_hex = block_data[pos:pos + 7].decode('ascii')
        pos += 7

        # Event type
        event_type = block_data[pos:pos + 16].rstrip(b'\x00').decode('utf-8')
        pos += 16

        # Significance
        significance = struct.unpack('!f', block_data[pos:pos + 4])[0]
        pos += 4

        # Narrative
        narrative_len = struct.unpack('!H', block_data[pos:pos + 2])[0]
        pos += 2
        narrative = block_data[pos:pos + narrative_len - 1].decode('utf-8')
        pos += narrative_len

        # Block hash
        block_hash = block_data[pos:pos + 64].decode('ascii')

        # Reconstruct lattice coords
        lattice_coords = struct.unpack('!5f', lattice_compact)

        # Create color state from hex
        color_state = ColorState.from_hex(color_hex)

        # Create GradientBloomEvent
        bloom = GradientBloomEvent(
            event_id=block_hash[:16],
            agent_id=agent_id,
            timestamp=timestamp,
            event_type=event_type,
            significance=significance,
            narrative_fragment=narrative,
            color_state=color_state,
            lattice_coords=lattice_coords,
            dominant_archetype=archetype,
            corridor_id=corridor,
            primitive_signature=signature,
        )

        # Create block
        block = ColorEncodedBlock(bloom, prev_hash)
        block.timestamp = timestamp
        block.hash = block_hash

        blocks.append(block)

    return blocks


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Core classes
    'ColorEncodedBlock',
    'GradientBloomEvent',

    # ID mappings
    'ARCHETYPE_TO_ID',
    'ID_TO_ARCHETYPE',

    # Color mappings
    'CORRIDOR_COLORS',
    'ARCHETYPE_VISUALIZATION_GLYPHS',

    # Helper functions
    'create_color_block_from_bloom',
    'encode_block_chain',
    'decode_block_chain',
    'verify_color_chain',
]
