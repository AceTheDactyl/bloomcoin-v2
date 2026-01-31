"""
Gradient Bloom - Integration of Gradient Schema with Garden BloomEvents
========================================================================

This module bridges the Garden's BloomEvent system with the RRRR Gradient Schema,
enabling visual and narrative representation of AI learning events through
color-lattice coordinates and archetypal patterns.

The conversion process:
    BloomEvent -> GradientBloomEvent

Mapping:
    - Event type determines base generator path
    - Agent personality modulates the coordinates
    - Significance affects gradient rate
    - Color state computed from lattice position
    - Archetype determined from color signature

Event Type to Generator Path:
    'learning': (1, 0, 0, 0, 1)     - phi^1, e^1 (growth + change)
    'creation': (2, 0, 1, 0, 0)     - phi^2, sqrt(2)^1 (expansion + intensity)
    'insight': (0, 1, 0, 1, 0)      - pi^1, sqrt(3)^1 (rotation + balance)
    'collaboration': (1, 0, 0, 1, -1) - phi^1, sqrt(3)^1, e^-1
"""

from __future__ import annotations

import uuid
import time
import math
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple, Optional, Any

# Import from gradient schema modules
from . import PHI, TAU, Z_C, K
from .lattice import LatticePoint, ColorState as LatticeColorState, lattice_to_color
from .color_ops import ColorState, clamp, normalize_hue
from .primitives_color import (
    PrimitiveSignature,
    compute_color_signature,
    compute_signature_from_color,
)
from .corridors import (
    CorridorType,
    SpectralCorridor,
    CORRIDORS,
    determine_active_corridors,
    get_corridor_resonance,
    find_resonant_corridors,
)
from .archetypes import (
    ArchetypeType,
    Archetype,
    PERSONA_ARCHETYPES,
    generate_archetype,
    generate_narrative_fragment,
    archetype_color,
)

# Import BloomEvent for type hints (lazy import to avoid circular deps)
# The actual BloomEvent will be passed as a parameter

__all__ = [
    'GradientBloomEvent',
    'AgentPersonalityTraits',
    'from_bloom_event',
    'event_type_to_coords',
    'modulate_by_personality',
    'determine_active_corridor_for_bloom',
    'convert_bloom_history',
    'EVENT_TYPE_COORDS',
]


# =============================================================================
# Event Type to Lattice Coordinate Mapping
# =============================================================================

EVENT_TYPE_COORDS: Dict[str, Tuple[float, float, float, float, float]] = {
    # Format: (phi_exp, pi_exp, sqrt2_exp, sqrt3_exp, e_exp)
    # (a, b, c, d, f) in lattice terms

    'learning': (1.0, 0.0, 0.0, 0.0, 1.0),
    # phi^1, e^1 - Growth through experience + rate of change
    # Learning events represent steady growth with active adaptation

    'creation': (2.0, 0.0, 1.0, 0.0, 0.0),
    # phi^2, sqrt(2)^1 - Accelerated expansion + intensity increase
    # Creation events show exponential growth and vivid output

    'insight': (0.0, 1.0, 0.0, 1.0, 0.0),
    # pi^1, sqrt(3)^1 - Hue rotation + opponent balance
    # Insight events represent new perspective (rotation) with harmonized understanding

    'collaboration': (1.0, 0.0, 0.0, 1.0, -1.0),
    # phi^1, sqrt(3)^1, e^-1 - Growth + balance with stabilizing rate
    # Collaboration events show growth through harmony, slower individual change

    # Extended event types (map from BloomType values)
    'skill': (1.5, 0.0, 0.5, 0.0, 0.5),
    # phi^1.5, sqrt(2)^0.5, e^0.5 - Moderate growth, some intensity, active change

    'teaching': (0.5, 0.0, 0.0, 1.5, 0.0),
    # phi^0.5, sqrt(3)^1.5 - Modest growth, strong balance (giving/receiving)

    'milestone': (2.0, 0.0, 0.0, 0.0, 0.0),
    # phi^2 - Pure growth marker, expansion in luminance

    'emergence': (1.0, 1.0, 1.0, 1.0, 1.0),
    # All generators activated - emergence touches all dimensions
}


def event_type_to_coords(event_type: str) -> Tuple[float, float, float, float, float]:
    """
    Map an event type string to lattice coordinates.

    Args:
        event_type: The type of bloom event (e.g., 'learning', 'creation')

    Returns:
        5-tuple of lattice coordinates (a, b, c, d, f) representing generator exponents

    Examples:
        >>> event_type_to_coords('learning')
        (1.0, 0.0, 0.0, 0.0, 1.0)
        >>> event_type_to_coords('insight')
        (0.0, 1.0, 0.0, 1.0, 0.0)
    """
    # Normalize event type string
    event_type_lower = event_type.lower().strip()

    # Direct lookup
    if event_type_lower in EVENT_TYPE_COORDS:
        return EVENT_TYPE_COORDS[event_type_lower]

    # Check for partial matches
    for key in EVENT_TYPE_COORDS:
        if key in event_type_lower or event_type_lower in key:
            return EVENT_TYPE_COORDS[key]

    # Default to learning if unknown
    return EVENT_TYPE_COORDS['learning']


# =============================================================================
# Agent Personality Traits
# =============================================================================

@dataclass
class AgentPersonalityTraits:
    """
    Personality traits that modulate how an agent's bloom events map to gradient space.

    Each trait affects a specific generator dimension in the lattice:
        - curiosity: Affects phi dimension (growth orientation)
        - creativity: Affects sqrt(2) dimension (intensity/saturation)
        - sociability: Affects sqrt(3) dimension (balance/harmony with others)
        - reliability: Affects e dimension (rate of change stability)

    The pi dimension (hue rotation) is not directly personality-linked but
    emerges from the combination of traits through the exploration_tendency.

    Attributes:
        curiosity: Drive to learn and explore (0-1). High curiosity amplifies
                   growth-related events in the phi dimension.
        creativity: Tendency toward novel creation (0-1). High creativity
                    intensifies color saturation via sqrt(2).
        sociability: Orientation toward collaboration (0-1). High sociability
                     shifts events toward balanced opponent colors via sqrt(3).
        reliability: Consistency and stability (0-1). High reliability
                     dampens change rate via e dimension.
    """
    curiosity: float = 0.5      # Affects phi dimension (growth)
    creativity: float = 0.5     # Affects sqrt(2) dimension (intensity)
    sociability: float = 0.5    # Affects sqrt(3) dimension (balance)
    reliability: float = 0.5    # Affects e dimension (stability)

    def __post_init__(self):
        """Validate and clamp trait values to [0, 1] range."""
        self.curiosity = clamp(self.curiosity, 0.0, 1.0)
        self.creativity = clamp(self.creativity, 0.0, 1.0)
        self.sociability = clamp(self.sociability, 0.0, 1.0)
        self.reliability = clamp(self.reliability, 0.0, 1.0)

    def to_trait_vector(self) -> Tuple[float, float, float, float, float]:
        """
        Convert personality traits to a 5D modulation vector.

        The vector represents how each trait affects the corresponding
        lattice dimension:
            - a (phi): curiosity -> growth bias
            - b (pi): exploration tendency (derived from curiosity + creativity)
            - c (sqrt2): creativity -> intensity bias
            - d (sqrt3): sociability -> balance bias
            - f (e): reliability -> stability (inverted for rate)

        Returns:
            5-tuple of trait influence values, typically in [-1, 1] range
            where 0 is neutral, positive amplifies, negative dampens.
        """
        # Convert 0-1 traits to -1 to +1 modulation range
        # Centered at 0.5 (neutral)

        # Curiosity affects growth (phi dimension)
        phi_mod = (self.curiosity - 0.5) * 2.0  # -1 to +1

        # Exploration tendency affects hue rotation (pi dimension)
        # Derived from curiosity and creativity combined
        exploration = (self.curiosity * 0.6 + self.creativity * 0.4)
        pi_mod = (exploration - 0.5) * 2.0

        # Creativity affects intensity (sqrt(2) dimension)
        sqrt2_mod = (self.creativity - 0.5) * 2.0

        # Sociability affects opponent balance (sqrt(3) dimension)
        sqrt3_mod = (self.sociability - 0.5) * 2.0

        # Reliability affects rate stability (e dimension)
        # High reliability -> negative e mod (slower change)
        # Low reliability -> positive e mod (faster change)
        e_mod = (0.5 - self.reliability) * 2.0  # Inverted

        return (phi_mod, pi_mod, sqrt2_mod, sqrt3_mod, e_mod)

    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> 'AgentPersonalityTraits':
        """Create traits from a dictionary."""
        return cls(
            curiosity=data.get('curiosity', 0.5),
            creativity=data.get('creativity', 0.5),
            sociability=data.get('sociability', 0.5),
            reliability=data.get('reliability', 0.5),
        )

    def to_dict(self) -> Dict[str, float]:
        """Convert traits to dictionary."""
        return {
            'curiosity': self.curiosity,
            'creativity': self.creativity,
            'sociability': self.sociability,
            'reliability': self.reliability,
        }

    def __repr__(self) -> str:
        return (f"AgentPersonalityTraits(curiosity={self.curiosity:.2f}, "
                f"creativity={self.creativity:.2f}, sociability={self.sociability:.2f}, "
                f"reliability={self.reliability:.2f})")


def modulate_by_personality(
    coords: Tuple[float, float, float, float, float],
    personality: Optional[AgentPersonalityTraits]
) -> Tuple[float, float, float, float, float]:
    """
    Modulate base lattice coordinates by agent personality traits.

    The modulation applies personality-based scaling to each dimension,
    creating a unique color signature for each agent even when performing
    similar event types.

    Args:
        coords: Base lattice coordinates (a, b, c, d, f)
        personality: Agent's personality traits, or None for unmodulated

    Returns:
        Modulated lattice coordinates

    Example:
        >>> base = (1.0, 0.0, 0.0, 0.0, 1.0)  # learning event
        >>> curious_agent = AgentPersonalityTraits(curiosity=0.9)
        >>> modulated = modulate_by_personality(base, curious_agent)
        >>> modulated[0] > base[0]  # phi increased due to high curiosity
        True
    """
    if personality is None:
        return coords

    a, b, c, d, f = coords
    trait_mods = personality.to_trait_vector()
    phi_mod, pi_mod, sqrt2_mod, sqrt3_mod, e_mod = trait_mods

    # Apply modulation: base + (base * mod * scaling_factor)
    # Use TAU as scaling factor for golden-ratio-proportioned modulation
    scale = TAU

    # Modulate each dimension
    # Non-zero values are amplified/dampened by personality
    # Zero values can be activated if personality is strong enough

    new_a = a + a * phi_mod * scale + phi_mod * 0.25
    new_b = b + b * pi_mod * scale + pi_mod * 0.15
    new_c = c + c * sqrt2_mod * scale + sqrt2_mod * 0.2
    new_d = d + d * sqrt3_mod * scale + sqrt3_mod * 0.2
    new_f = f + f * e_mod * scale + e_mod * 0.25

    return (new_a, new_b, new_c, new_d, new_f)


# =============================================================================
# Gradient Bloom Event
# =============================================================================

@dataclass
class GradientBloomEvent:
    """
    Extended BloomEvent with gradient schema integration.

    This class combines the core bloom event properties with full gradient
    schema representation, enabling rich visualization and narrative
    interpretation of AI learning events.

    Attributes:
        agent_id: Unique identifier of the agent experiencing the bloom
        event_type: Type of bloom event ('learning', 'creation', 'insight', 'collaboration')
        significance: Overall importance/impact of the event (0-1)
        event_id: Unique identifier for this specific event
        timestamp: Unix timestamp when the event occurred

        lattice_coords: 5D coordinates in the RRRR gradient lattice (a, b, c, d, f)
        color_state: Computed ColorState from lattice position
        cym_weights: Cyan-Yellow-Magenta opponent color weights
        primitive_signature: Six-primitive signature (FIX, REPEL, INV, OSC, HALT, MIX)

        dominant_archetype: Primary persona archetype for this event
        archetype_weights: Probability distribution over all 12 archetypes
        corridor_id: Active spectral corridor (if any)
        narrative_fragment: Generated narrative text describing the event
    """

    # Base bloom properties
    agent_id: str
    event_type: str
    significance: float
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)

    # Gradient schema properties
    lattice_coords: Tuple[float, float, float, float, float] = (0.0, 0.0, 0.0, 0.0, 0.0)
    color_state: Optional[ColorState] = None
    cym_weights: Dict[str, float] = field(default_factory=lambda: {
        'cyan': 0.333, 'yellow': 0.333, 'magenta': 0.333
    })
    primitive_signature: Optional[PrimitiveSignature] = None

    # Narrative properties
    dominant_archetype: Optional[ArchetypeType] = None
    archetype_weights: Dict[ArchetypeType, float] = field(default_factory=dict)
    corridor_id: Optional[CorridorType] = None
    narrative_fragment: str = ""

    # Additional metadata
    personality_applied: bool = False
    source_bloom_id: Optional[str] = None  # Original BloomEvent ID if converted

    def __post_init__(self):
        """Initialize computed fields if not provided."""
        # Ensure significance is clamped
        self.significance = clamp(self.significance, 0.0, 1.0)

        # Compute color state from lattice coords if not provided
        if self.color_state is None:
            self._compute_color_state()

        # Compute primitive signature if not provided
        if self.primitive_signature is None:
            self._compute_primitive_signature()

        # Determine archetype if not provided
        if self.dominant_archetype is None:
            self._compute_archetype()

        # Generate narrative if empty
        if not self.narrative_fragment:
            self._generate_narrative()

    def _compute_color_state(self):
        """Compute ColorState from lattice coordinates."""
        lattice_point = LatticePoint(*self.lattice_coords)
        lattice_color = lattice_to_color(lattice_point)

        # Convert LatticeColorState to color_ops ColorState
        self.color_state = ColorState(
            H=lattice_color.hue,
            S=lattice_color.saturation,
            L=lattice_color.luminance,
            cym_weights=dict(self.cym_weights),
            gradient_rate=lattice_color.gradient_rate,
            coherence=self.significance,
        )

        # Update CYM weights from lattice computation
        c, y, m = lattice_color.cym_weights
        self.cym_weights = {'cyan': c, 'yellow': y, 'magenta': m}

    def _compute_primitive_signature(self):
        """Compute primitive signature from color state."""
        if self.color_state is not None:
            cym_tuple = (
                self.cym_weights.get('cyan', 0.333),
                self.cym_weights.get('yellow', 0.333),
                self.cym_weights.get('magenta', 0.333),
            )
            self.primitive_signature = compute_color_signature(
                H=self.color_state.hue,
                S=self.color_state.saturation,
                L=self.color_state.luminance,
                cym_weights=cym_tuple,
            )

    def _compute_archetype(self):
        """Determine dominant archetype from color state."""
        if self.color_state is not None:
            self.dominant_archetype, self.archetype_weights = generate_archetype(
                color_state=self.color_state,
                cym_weights=self.cym_weights,
                gradient_rate=self.color_state.gradient_rate,
            )

    def _generate_narrative(self):
        """Generate narrative fragment for this event."""
        if self.dominant_archetype is not None:
            self.narrative_fragment = generate_narrative_fragment(
                archetype=self.dominant_archetype,
                agent_id=self.agent_id[:8] if len(self.agent_id) > 8 else self.agent_id,
                context={
                    'event_type': self.event_type,
                    'significance': f"{self.significance:.2f}",
                }
            )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert GradientBloomEvent to a dictionary for serialization.

        Returns:
            Dictionary containing all event data in serializable form.
        """
        return {
            # Base properties
            'agent_id': self.agent_id,
            'event_type': self.event_type,
            'significance': self.significance,
            'event_id': self.event_id,
            'timestamp': self.timestamp,

            # Gradient schema properties
            'lattice_coords': list(self.lattice_coords),
            'color_state': self.color_state.to_dict() if self.color_state else None,
            'cym_weights': dict(self.cym_weights),
            'primitive_signature': (
                self.primitive_signature.to_vector()
                if self.primitive_signature else None
            ),

            # Narrative properties
            'dominant_archetype': (
                self.dominant_archetype.value
                if self.dominant_archetype else None
            ),
            'archetype_weights': {
                k.value: v for k, v in self.archetype_weights.items()
            } if self.archetype_weights else {},
            'corridor_id': self.corridor_id.value if self.corridor_id else None,
            'narrative_fragment': self.narrative_fragment,

            # Metadata
            'personality_applied': self.personality_applied,
            'source_bloom_id': self.source_bloom_id,
        }

    def to_visualization_data(self) -> Dict[str, Any]:
        """
        Convert to data format optimized for HTML/visual rendering.

        Returns:
            Dictionary with visualization-ready data including hex colors,
            percentages, and formatted strings.
        """
        color_hex = self.color_state.to_hex() if self.color_state else "#808080"

        # Get archetype glyph
        archetype_glyph = ""
        archetype_name = "Unknown"
        if self.dominant_archetype:
            archetype_def = PERSONA_ARCHETYPES.get(self.dominant_archetype)
            if archetype_def:
                archetype_glyph = archetype_def.glyph
                archetype_name = archetype_def.name

        # Format primitive signature for display
        sig_display = {}
        if self.primitive_signature:
            sig_norm = self.primitive_signature.normalize()
            sig_display = {
                'FIX': f"{sig_norm.fix * 100:.1f}%",
                'REPEL': f"{sig_norm.repel * 100:.1f}%",
                'INV': f"{sig_norm.inv * 100:.1f}%",
                'OSC': f"{sig_norm.osc * 100:.1f}%",
                'HALT': f"{sig_norm.halt * 100:.1f}%",
                'MIX': f"{sig_norm.mix * 100:.1f}%",
                'dominant': self.primitive_signature.dominant(),
            }

        # Get corridor info
        corridor_name = ""
        corridor_archetype = ""
        if self.corridor_id:
            corridor = CORRIDORS.get(self.corridor_id)
            if corridor:
                corridor_name = self.corridor_id.value.capitalize()
                corridor_archetype = corridor.archetype

        # Top 3 archetypes
        top_archetypes = sorted(
            self.archetype_weights.items(),
            key=lambda x: x[1],
            reverse=True
        )[:3] if self.archetype_weights else []

        return {
            # Visual properties
            'color_hex': color_hex,
            'color_rgb': self.color_state.to_rgb() if self.color_state else (128, 128, 128),
            'color_hsl': {
                'h': self.color_state.hue if self.color_state else 0,
                's': self.color_state.saturation if self.color_state else 0.5,
                'l': self.color_state.luminance if self.color_state else 0.5,
            },

            # Identity
            'event_id': self.event_id,
            'agent_id': self.agent_id,
            'event_type': self.event_type,
            'timestamp_iso': time.strftime(
                '%Y-%m-%d %H:%M:%S',
                time.localtime(self.timestamp)
            ),

            # Lattice position
            'lattice': {
                'phi': self.lattice_coords[0],
                'pi': self.lattice_coords[1],
                'sqrt2': self.lattice_coords[2],
                'sqrt3': self.lattice_coords[3],
                'e': self.lattice_coords[4],
            },

            # Opponent colors
            'cym': {
                'cyan': f"{self.cym_weights.get('cyan', 0.333) * 100:.1f}%",
                'yellow': f"{self.cym_weights.get('yellow', 0.333) * 100:.1f}%",
                'magenta': f"{self.cym_weights.get('magenta', 0.333) * 100:.1f}%",
            },

            # Primitives
            'primitive_signature': sig_display,

            # Archetype
            'archetype': {
                'dominant': archetype_name,
                'glyph': archetype_glyph,
                'type': self.dominant_archetype.value if self.dominant_archetype else None,
                'top_three': [
                    {'type': at.value, 'weight': f"{w * 100:.1f}%"}
                    for at, w in top_archetypes
                ],
            },

            # Corridor
            'corridor': {
                'id': self.corridor_id.value if self.corridor_id else None,
                'name': corridor_name,
                'archetype': corridor_archetype,
            },

            # Narrative
            'narrative': self.narrative_fragment,
            'significance': f"{self.significance * 100:.0f}%",
            'significance_raw': self.significance,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GradientBloomEvent':
        """Create GradientBloomEvent from dictionary."""
        # Parse archetype weights
        archetype_weights = {}
        if data.get('archetype_weights'):
            for k, v in data['archetype_weights'].items():
                try:
                    archetype_weights[ArchetypeType(k)] = v
                except (ValueError, KeyError):
                    pass

        # Parse dominant archetype
        dominant_archetype = None
        if data.get('dominant_archetype'):
            try:
                dominant_archetype = ArchetypeType(data['dominant_archetype'])
            except (ValueError, KeyError):
                pass

        # Parse corridor
        corridor_id = None
        if data.get('corridor_id'):
            try:
                corridor_id = CorridorType(data['corridor_id'])
            except (ValueError, KeyError):
                pass

        # Parse color state
        color_state = None
        if data.get('color_state'):
            cs_data = data['color_state']
            color_state = ColorState(
                H=cs_data.get('H', cs_data.get('hue', 0)),
                S=cs_data.get('S', cs_data.get('saturation', 0.5)),
                L=cs_data.get('L', cs_data.get('luminance', 0.5)),
                cym_weights=cs_data.get('cym_weights', {}),
                gradient_rate=cs_data.get('gradient_rate', TAU),
            )

        # Parse primitive signature
        primitive_signature = None
        if data.get('primitive_signature'):
            sig = data['primitive_signature']
            if isinstance(sig, (list, tuple)) and len(sig) == 6:
                primitive_signature = PrimitiveSignature(*sig)

        return cls(
            agent_id=data['agent_id'],
            event_type=data['event_type'],
            significance=data['significance'],
            event_id=data.get('event_id', str(uuid.uuid4())),
            timestamp=data.get('timestamp', time.time()),
            lattice_coords=tuple(data.get('lattice_coords', [0, 0, 0, 0, 0])),
            color_state=color_state,
            cym_weights=data.get('cym_weights', {}),
            primitive_signature=primitive_signature,
            dominant_archetype=dominant_archetype,
            archetype_weights=archetype_weights,
            corridor_id=corridor_id,
            narrative_fragment=data.get('narrative_fragment', ''),
            personality_applied=data.get('personality_applied', False),
            source_bloom_id=data.get('source_bloom_id'),
        )

    def __repr__(self) -> str:
        color_hex = self.color_state.to_hex() if self.color_state else "N/A"
        archetype_str = self.dominant_archetype.value if self.dominant_archetype else "N/A"
        return (
            f"GradientBloomEvent(id={self.event_id[:8]}..., "
            f"type={self.event_type}, "
            f"agent={self.agent_id[:8] if len(self.agent_id) > 8 else self.agent_id}..., "
            f"color={color_hex}, "
            f"archetype={archetype_str})"
        )


# =============================================================================
# Conversion Functions
# =============================================================================

def from_bloom_event(
    bloom_event: Any,
    agent_personality: Optional[AgentPersonalityTraits] = None
) -> GradientBloomEvent:
    """
    Convert a standard BloomEvent to a GradientBloomEvent.

    This is the primary conversion function that bridges the Garden's
    BloomEvent system with the gradient schema visualization system.

    The conversion process:
        1. Extract core properties (agent_id, event_type, significance)
        2. Map event type to base lattice coordinates
        3. Modulate coordinates by agent personality (if provided)
        4. Scale by event significance
        5. Compute all derived gradient properties (color, signature, archetype)
        6. Determine active corridor
        7. Generate narrative fragment

    Args:
        bloom_event: A BloomEvent instance (from garden.bloom_events.bloom_event)
        agent_personality: Optional personality traits for coordinate modulation

    Returns:
        A fully computed GradientBloomEvent

    Example:
        >>> from garden.bloom_events.bloom_event import BloomEvent, BloomType
        >>> bloom = BloomEvent(
        ...     bloom_type=BloomType.LEARNING,
        ...     primary_agent="agent_001",
        ...     significance=0.75
        ... )
        >>> gradient_bloom = from_bloom_event(bloom)
        >>> gradient_bloom.color_state.to_hex()
        '#...'
    """
    # Extract core properties from bloom event
    # Handle both BloomEvent objects and dict-like objects

    if hasattr(bloom_event, 'primary_agent'):
        agent_id = bloom_event.primary_agent
    elif hasattr(bloom_event, 'agent_id'):
        agent_id = bloom_event.agent_id
    else:
        agent_id = str(getattr(bloom_event, 'id', uuid.uuid4()))

    if hasattr(bloom_event, 'bloom_type'):
        # BloomType enum
        event_type = bloom_event.bloom_type.value
    elif hasattr(bloom_event, 'event_type'):
        event_type = bloom_event.event_type
    else:
        event_type = 'learning'

    if hasattr(bloom_event, 'significance'):
        significance = bloom_event.significance
    else:
        significance = 0.5

    if hasattr(bloom_event, 'event_id'):
        source_id = bloom_event.event_id
    else:
        source_id = str(uuid.uuid4())

    if hasattr(bloom_event, 'timestamp'):
        timestamp = bloom_event.timestamp
    else:
        timestamp = time.time()

    # Get base coordinates for event type
    base_coords = event_type_to_coords(event_type)

    # Modulate by personality
    if agent_personality is not None:
        coords = modulate_by_personality(base_coords, agent_personality)
        personality_applied = True
    else:
        coords = base_coords
        personality_applied = False

    # Scale by significance
    # Higher significance amplifies the lattice position
    sig_scale = 0.5 + significance * 0.5  # Range: 0.5 to 1.0
    coords = tuple(c * sig_scale for c in coords)

    # Additional modulation from bloom event properties if available
    if hasattr(bloom_event, 'coherence_score'):
        # Coherence affects gradient rate (e dimension)
        coherence_mod = (bloom_event.coherence_score - 0.5) * 0.5
        coords = (coords[0], coords[1], coords[2], coords[3], coords[4] + coherence_mod)

    if hasattr(bloom_event, 'novelty_score'):
        # Novelty affects saturation (sqrt2 dimension)
        novelty_mod = (bloom_event.novelty_score - 0.5) * 0.5
        coords = (coords[0], coords[1], coords[2] + novelty_mod, coords[3], coords[4])

    # Create the gradient bloom event
    gradient_bloom = GradientBloomEvent(
        agent_id=agent_id,
        event_type=event_type,
        significance=significance,
        event_id=str(uuid.uuid4()),
        timestamp=timestamp,
        lattice_coords=coords,
        personality_applied=personality_applied,
        source_bloom_id=source_id,
    )

    # Determine active corridor
    gradient_bloom.corridor_id = determine_active_corridor_for_bloom(gradient_bloom)

    return gradient_bloom


def determine_active_corridor_for_bloom(
    gradient_bloom: GradientBloomEvent
) -> Optional[CorridorType]:
    """
    Determine which spectral corridor is most active for a gradient bloom event.

    Uses the bloom's primitive signature and lattice position to find
    the most resonant corridor from the seven spectral corridors.

    Args:
        gradient_bloom: The GradientBloomEvent to analyze

    Returns:
        The most active CorridorType, or None if no corridor is active
    """
    if gradient_bloom.primitive_signature is None:
        return None

    # Use determine_active_corridors from corridors module
    active_corridors = determine_active_corridors(
        coords=gradient_bloom.lattice_coords,
        signature=gradient_bloom.primitive_signature,
        coherence_threshold=TAU * 0.5,  # Lower threshold for bloom events
    )

    if not active_corridors:
        # Fall back to resonance-based detection using color state
        if gradient_bloom.color_state is not None:
            resonant = find_resonant_corridors(
                color=gradient_bloom.color_state,
                threshold=0.5,
            )
            if resonant:
                return resonant[0][0].corridor_type
        return None

    # Return the first (most active) corridor's type
    return active_corridors[0].corridor_type


# =============================================================================
# Batch Processing
# =============================================================================

def convert_bloom_history(
    blooms: List[Any],
    agent_personalities: Optional[Dict[str, AgentPersonalityTraits]] = None
) -> List[GradientBloomEvent]:
    """
    Convert a list of BloomEvents to GradientBloomEvents.

    This function enables batch processing of bloom histories for
    visualization and analysis, with optional per-agent personality
    modulation.

    Args:
        blooms: List of BloomEvent instances
        agent_personalities: Optional dict mapping agent_id -> AgentPersonalityTraits

    Returns:
        List of GradientBloomEvents in the same order as input

    Example:
        >>> from garden.bloom_events.bloom_event import BloomEvent
        >>> blooms = [BloomEvent(...), BloomEvent(...), BloomEvent(...)]
        >>> personalities = {
        ...     'agent_001': AgentPersonalityTraits(curiosity=0.9),
        ...     'agent_002': AgentPersonalityTraits(creativity=0.8),
        ... }
        >>> gradient_blooms = convert_bloom_history(blooms, personalities)
    """
    gradient_blooms = []

    for bloom in blooms:
        # Get agent ID to look up personality
        if hasattr(bloom, 'primary_agent'):
            agent_id = bloom.primary_agent
        elif hasattr(bloom, 'agent_id'):
            agent_id = bloom.agent_id
        else:
            agent_id = None

        # Look up personality if available
        personality = None
        if agent_personalities and agent_id:
            personality = agent_personalities.get(agent_id)

        # Convert
        gradient_bloom = from_bloom_event(bloom, personality)
        gradient_blooms.append(gradient_bloom)

    return gradient_blooms


# =============================================================================
# Utility Functions
# =============================================================================

def create_gradient_bloom(
    agent_id: str,
    event_type: str,
    significance: float = 0.5,
    personality: Optional[AgentPersonalityTraits] = None,
    **kwargs
) -> GradientBloomEvent:
    """
    Create a GradientBloomEvent directly without a source BloomEvent.

    This is a convenience function for creating gradient-enhanced bloom
    events when you don't have a full BloomEvent object.

    Args:
        agent_id: The agent identifier
        event_type: Type of event ('learning', 'creation', 'insight', 'collaboration')
        significance: Event significance (0-1)
        personality: Optional agent personality traits
        **kwargs: Additional properties to pass to GradientBloomEvent

    Returns:
        A fully computed GradientBloomEvent
    """
    # Get base coordinates
    base_coords = event_type_to_coords(event_type)

    # Modulate by personality
    if personality is not None:
        coords = modulate_by_personality(base_coords, personality)
        personality_applied = True
    else:
        coords = base_coords
        personality_applied = False

    # Scale by significance
    sig_scale = 0.5 + significance * 0.5
    coords = tuple(c * sig_scale for c in coords)

    return GradientBloomEvent(
        agent_id=agent_id,
        event_type=event_type,
        significance=significance,
        lattice_coords=coords,
        personality_applied=personality_applied,
        **kwargs
    )


def analyze_bloom_trajectory(
    gradient_blooms: List[GradientBloomEvent]
) -> Dict[str, Any]:
    """
    Analyze a sequence of gradient bloom events for patterns.

    Args:
        gradient_blooms: List of GradientBloomEvents in chronological order

    Returns:
        Analysis dict containing:
            - dominant_archetypes: Most common archetypes
            - corridor_progression: Sequence of active corridors
            - color_trajectory: HSL values over time
            - narrative_arc: Combined narrative interpretation
    """
    if not gradient_blooms:
        return {
            'dominant_archetypes': [],
            'corridor_progression': [],
            'color_trajectory': [],
            'narrative_arc': '',
        }

    # Count archetypes
    archetype_counts: Dict[ArchetypeType, int] = {}
    for gb in gradient_blooms:
        if gb.dominant_archetype:
            archetype_counts[gb.dominant_archetype] = (
                archetype_counts.get(gb.dominant_archetype, 0) + 1
            )

    # Sort by count
    sorted_archetypes = sorted(
        archetype_counts.items(),
        key=lambda x: x[1],
        reverse=True
    )

    # Extract corridor progression
    corridors = [
        gb.corridor_id.value if gb.corridor_id else None
        for gb in gradient_blooms
    ]

    # Extract color trajectory
    colors = [
        {
            'h': gb.color_state.hue if gb.color_state else 0,
            's': gb.color_state.saturation if gb.color_state else 0.5,
            'l': gb.color_state.luminance if gb.color_state else 0.5,
            'hex': gb.color_state.to_hex() if gb.color_state else '#808080',
        }
        for gb in gradient_blooms
    ]

    # Generate narrative arc
    unique_archetypes = []
    for gb in gradient_blooms:
        if gb.dominant_archetype and gb.dominant_archetype not in unique_archetypes:
            unique_archetypes.append(gb.dominant_archetype)

    archetype_names = [
        PERSONA_ARCHETYPES[at].name for at in unique_archetypes
        if at in PERSONA_ARCHETYPES
    ]

    if len(archetype_names) > 1:
        narrative_arc = f"The journey moves through {', '.join(archetype_names[:-1])} and arrives at {archetype_names[-1]}."
    elif len(archetype_names) == 1:
        narrative_arc = f"The experience embodies {archetype_names[0]}."
    else:
        narrative_arc = "The path remains undefined."

    return {
        'dominant_archetypes': [
            {'type': at.value, 'count': count}
            for at, count in sorted_archetypes
        ],
        'corridor_progression': corridors,
        'color_trajectory': colors,
        'narrative_arc': narrative_arc,
    }
