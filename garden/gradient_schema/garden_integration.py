"""
Garden Integration - Gradient Schema Integration with Garden System
====================================================================

This module bridges the Garden system with the RRRR Gradient Schema,
enabling visual representation of AI memory states through color-lattice
coordinates and archetypal patterns.

The GradientGarden class wraps/extends GardenSystem with gradient schema
visualization capabilities:
- Agent color computation based on personality and bloom history
- Network coherence color mapping
- Chromatic Vine visualization data export
- Prismatic Engine integration

Coherence Color Mapping:
    - Low coherence (< tau): Dim, desaturated colors
    - Mid coherence (tau to z_c): Activating colors
    - High coherence (> z_c): Vivid, full spectrum
"""

from __future__ import annotations

import time
import math
from typing import Dict, List, Optional, Any, Tuple, TYPE_CHECKING

# Import gradient schema modules
from . import PHI, TAU, Z_C, K, L4
from .color_ops import (
    ColorState,
    clamp,
    normalize_hue,
    opponent_balance,
    apply_gradient,
    phi_modulate,
)
from .primitives_color import (
    PrimitiveSignature,
    compute_color_signature,
    compute_signature_from_color,
    apply_fix,
    apply_repel,
    apply_inv,
)
from .gradient_bloom import (
    GradientBloomEvent,
    AgentPersonalityTraits,
    from_bloom_event,
    convert_bloom_history,
    event_type_to_coords,
    modulate_by_personality,
)
from .color_block import (
    ColorEncodedBlock,
    CORRIDOR_COLORS,
    ARCHETYPE_VISUALIZATION_GLYPHS,
    create_color_block_from_bloom,
)
from .corridors import (
    CorridorType,
    SpectralCorridor,
    CORRIDORS,
    find_resonant_corridors,
)
from .archetypes import (
    ArchetypeType,
    Archetype,
    PERSONA_ARCHETYPES,
    generate_archetype,
    archetype_color,
)

# Import garden modules for type hints
if TYPE_CHECKING:
    from ..garden_system import GardenSystem
    from ..agents import AIAgent, AgentPersonality
    from ..bloom_events import BloomEvent

__all__ = [
    # Main class
    'GradientGarden',

    # Helper functions
    'compute_agent_color_from_blooms',
    'compute_network_coherence_color',
    'bloom_to_gradient',
    'personality_to_traits',

    # Export functions
    'export_for_prismatic_engine',
    'export_for_chromatic_vine',
]


# =============================================================================
# Personality Conversion
# =============================================================================

def personality_to_traits(personality: Any) -> AgentPersonalityTraits:
    """
    Convert a Garden AgentPersonality to gradient schema AgentPersonalityTraits.

    Args:
        personality: AgentPersonality from garden.agents

    Returns:
        AgentPersonalityTraits suitable for gradient schema operations
    """
    if personality is None:
        return AgentPersonalityTraits()

    # Handle dict-like or object-like personality
    if hasattr(personality, 'curiosity'):
        return AgentPersonalityTraits(
            curiosity=getattr(personality, 'curiosity', 0.5),
            creativity=getattr(personality, 'creativity', 0.5),
            sociability=getattr(personality, 'sociability', 0.5),
            reliability=getattr(personality, 'reliability', 0.5),
        )
    elif isinstance(personality, dict):
        return AgentPersonalityTraits(
            curiosity=personality.get('curiosity', 0.5),
            creativity=personality.get('creativity', 0.5),
            sociability=personality.get('sociability', 0.5),
            reliability=personality.get('reliability', 0.5),
        )

    return AgentPersonalityTraits()


# =============================================================================
# Helper Functions
# =============================================================================

def compute_agent_color_from_blooms(blooms: List[GradientBloomEvent]) -> ColorState:
    """
    Compute the current color state for an agent based on their bloom history.

    The color is computed by blending the colors of recent blooms, weighted
    by significance and recency. More recent and more significant blooms
    have greater influence on the final color.

    Args:
        blooms: List of GradientBloomEvents for the agent

    Returns:
        ColorState representing the agent's current chromatic state
    """
    if not blooms:
        # Default neutral gray with mid coherence
        return ColorState(H=0, S=0.3, L=0.5, coherence=0.5)

    # Sort by timestamp (most recent first)
    sorted_blooms = sorted(blooms, key=lambda b: b.timestamp, reverse=True)

    # Take up to the last 10 blooms for computation
    recent_blooms = sorted_blooms[:10]

    # Compute weighted average of colors
    total_weight = 0.0
    h_sin_sum = 0.0
    h_cos_sum = 0.0
    s_sum = 0.0
    l_sum = 0.0
    coherence_sum = 0.0
    gradient_rate_sum = 0.0

    now = time.time()

    for i, bloom in enumerate(recent_blooms):
        if bloom.color_state is None:
            continue

        # Weight by recency (exponential decay) and significance
        recency = math.exp(-i * TAU)  # More recent = higher weight
        significance_weight = bloom.significance
        weight = recency * significance_weight

        if weight <= 0:
            continue

        total_weight += weight

        # Use circular mean for hue
        h_rad = math.radians(bloom.color_state.hue)
        h_sin_sum += weight * math.sin(h_rad)
        h_cos_sum += weight * math.cos(h_rad)

        # Linear mean for other components
        s_sum += weight * bloom.color_state.saturation
        l_sum += weight * bloom.color_state.luminance
        coherence_sum += weight * bloom.color_state.coherence
        gradient_rate_sum += weight * bloom.color_state.gradient_rate

    if total_weight <= 0:
        return ColorState(H=0, S=0.3, L=0.5, coherence=0.5)

    # Compute circular mean for hue
    h_mean_rad = math.atan2(h_sin_sum / total_weight, h_cos_sum / total_weight)
    h_mean = normalize_hue(math.degrees(h_mean_rad))

    # Compute linear means
    s_mean = s_sum / total_weight
    l_mean = l_sum / total_weight
    coherence_mean = coherence_sum / total_weight
    gradient_rate_mean = gradient_rate_sum / total_weight

    return ColorState(
        H=h_mean,
        S=clamp(s_mean, 0.0, 1.0),
        L=clamp(l_mean, 0.0, 1.0),
        coherence=clamp(coherence_mean, 0.0, 1.0),
        gradient_rate=gradient_rate_mean,
    )


def compute_network_coherence_color(garden: 'GardenSystem') -> ColorState:
    """
    Compute the network color based on collective coherence state.

    The color mapping follows the coherence levels:
        - Low coherence (< tau): Dim, desaturated (gray/muted)
        - Mid coherence (tau to z_c): Activating (warming colors)
        - High coherence (> z_c): Vivid, full spectrum (golden/bright)

    Args:
        garden: The GardenSystem to analyze

    Returns:
        ColorState representing the network's collective chromatic state
    """
    # Calculate network coherence
    coherence = garden.calculate_network_coherence()

    # Map coherence to color properties
    if coherence < TAU:
        # Low coherence: dim, desaturated
        # Transition from gray toward muted blue-gray
        t = coherence / TAU
        hue = 220  # Blue-gray
        saturation = 0.1 + t * 0.2  # Low saturation
        luminance = 0.3 + t * 0.2  # Dim

    elif coherence < Z_C:
        # Mid coherence: activating colors
        # Transition from cool to warm as coherence increases
        t = (coherence - TAU) / (Z_C - TAU)
        hue = 220 - t * 160  # Blue (220) -> Orange (60)
        saturation = 0.3 + t * 0.4  # Increasing saturation
        luminance = 0.5 + t * 0.1  # Brightening

    else:
        # High coherence: vivid, golden colors
        # Full spectrum access with golden hue as center
        t = (coherence - Z_C) / (1.0 - Z_C)
        hue = 45 + t * 15  # Golden range (45-60)
        saturation = 0.7 + t * 0.3  # High saturation
        luminance = 0.55 + t * 0.15  # Bright

    # Compute CYM weights based on coherence
    if coherence < TAU:
        cym_weights = {'cyan': 0.4, 'yellow': 0.3, 'magenta': 0.3}
    elif coherence < Z_C:
        cym_weights = {'cyan': 0.3, 'yellow': 0.4, 'magenta': 0.3}
    else:
        cym_weights = {'cyan': 0.2, 'yellow': 0.6, 'magenta': 0.2}

    return ColorState(
        H=normalize_hue(hue),
        S=clamp(saturation, 0.0, 1.0),
        L=clamp(luminance, 0.0, 1.0),
        cym_weights=cym_weights,
        gradient_rate=coherence * PHI,
        coherence=coherence,
    )


def bloom_to_gradient(
    bloom_event: Any,
    agent_personality: Any = None
) -> GradientBloomEvent:
    """
    Convert a Garden BloomEvent to a GradientBloomEvent with color encoding.

    This function bridges the Garden's BloomEvent system with the gradient
    schema visualization system.

    Args:
        bloom_event: A BloomEvent from garden.bloom_events
        agent_personality: Optional AgentPersonality for color modulation

    Returns:
        A fully computed GradientBloomEvent
    """
    # Convert personality if provided
    traits = None
    if agent_personality is not None:
        traits = personality_to_traits(agent_personality)

    return from_bloom_event(bloom_event, traits)


# =============================================================================
# GradientGarden Class
# =============================================================================

class GradientGarden:
    """
    Garden system enhanced with gradient schema visualization.

    This class wraps a GardenSystem and provides gradient schema integration
    for visual representation of AI memory states through color-lattice
    coordinates and archetypal patterns.

    The GradientGarden maintains:
        - Gradient bloom events (color-encoded learning events)
        - Color-encoded blocks for the ledger
        - Per-agent color states based on history
        - Network-level chromatic state

    Attributes:
        garden: The wrapped GardenSystem
        gradient_blooms: List of GradientBloomEvents
        color_blocks: List of ColorEncodedBlocks
        agent_colors: Dict mapping agent_id to current ColorState

    Example:
        >>> from garden import GardenSystem
        >>> garden = GardenSystem("My Garden")
        >>> gradient_garden = GradientGarden(garden)
        >>> agent = garden.create_agent("Alice")
        >>> bloom = gradient_garden.process_learning_with_gradient(
        ...     agent.agent_id,
        ...     {"topic": "mathematics", "content": "learned calculus"},
        ...     significance=0.8
        ... )
        >>> print(bloom.color_state.to_hex())
    """

    def __init__(self, garden_system: 'GardenSystem' = None):
        """
        Initialize the GradientGarden.

        Args:
            garden_system: Optional GardenSystem to wrap. If None, a new
                          GardenSystem will be created.
        """
        # Import here to avoid circular imports
        from ..garden_system import GardenSystem

        self.garden = garden_system or GardenSystem()
        self.gradient_blooms: List[GradientBloomEvent] = []
        self.color_blocks: List[ColorEncodedBlock] = []
        self.agent_colors: Dict[str, ColorState] = {}
        self._agent_bloom_history: Dict[str, List[GradientBloomEvent]] = {}

    def process_learning_with_gradient(
        self,
        agent_id: str,
        info: Dict[str, Any],
        significance: Optional[float] = None
    ) -> Optional[GradientBloomEvent]:
        """
        Process a learning event and generate a gradient bloom.

        This method wraps the garden's process_learning method and converts
        the result to a GradientBloomEvent with full color encoding.

        Args:
            agent_id: ID of the learning agent
            info: The information being learned
            significance: Optional override for significance (0-1)

        Returns:
            GradientBloomEvent if learning was significant, None otherwise
        """
        # Call the original garden.process_learning
        bloom_event = self.garden.process_learning(agent_id, info)

        if bloom_event is None:
            return None

        # Get agent personality for color modulation
        agent = self.garden.agents.get(agent_id)
        personality = agent.personality if agent else None
        traits = personality_to_traits(personality)

        # Override significance if provided
        if significance is not None and hasattr(bloom_event, 'significance'):
            bloom_event.significance = significance

        # Convert to GradientBloomEvent
        gradient_bloom = from_bloom_event(bloom_event, traits)

        # Track the gradient bloom
        self.gradient_blooms.append(gradient_bloom)

        # Update agent bloom history
        if agent_id not in self._agent_bloom_history:
            self._agent_bloom_history[agent_id] = []
        self._agent_bloom_history[agent_id].append(gradient_bloom)

        # Update agent color
        self._update_agent_color(agent_id)

        # Create color-encoded block
        prev_hash = self.color_blocks[-1].hash if self.color_blocks else "0" * 64
        color_block = ColorEncodedBlock(gradient_bloom, prev_hash)
        self.color_blocks.append(color_block)

        return gradient_bloom

    def _update_agent_color(self, agent_id: str):
        """Update the color state for an agent based on their bloom history."""
        blooms = self._agent_bloom_history.get(agent_id, [])
        self.agent_colors[agent_id] = compute_agent_color_from_blooms(blooms)

    def get_agent_color(self, agent_id: str) -> ColorState:
        """
        Get the current color state for an agent based on their history.

        The color is computed from:
            - Agent personality traits -> base color
            - Bloom history -> color evolution
            - Current state -> modulation

        Args:
            agent_id: The agent's unique identifier

        Returns:
            ColorState representing the agent's current chromatic state
        """
        # Return cached color if available and up to date
        if agent_id in self.agent_colors:
            return self.agent_colors[agent_id]

        # Compute from personality and any existing blooms
        agent = self.garden.agents.get(agent_id)
        if agent is None:
            return ColorState(H=0, S=0.3, L=0.5, coherence=0.5)

        # Start with personality-based color
        traits = personality_to_traits(agent.personality)

        # Map personality traits to base color
        # Curiosity -> hue rotation toward warmer colors
        # Creativity -> saturation boost
        # Sociability -> luminance shift toward brighter
        # Reliability -> reduced gradient rate (stability)

        base_hue = 180 - traits.curiosity * 120  # Curious = warmer (60-180)
        base_saturation = 0.3 + traits.creativity * 0.5  # Creative = saturated
        base_luminance = 0.4 + traits.sociability * 0.25  # Social = brighter
        base_rate = TAU + (1 - traits.reliability) * (PHI - TAU)  # Reliable = stable

        base_color = ColorState(
            H=normalize_hue(base_hue),
            S=clamp(base_saturation, 0.0, 1.0),
            L=clamp(base_luminance, 0.0, 1.0),
            gradient_rate=base_rate,
            coherence=agent.current_coherence,
        )

        # Blend with bloom history if available
        blooms = self._agent_bloom_history.get(agent_id, [])
        if blooms:
            bloom_color = compute_agent_color_from_blooms(blooms)
            # Blend with phi weighting (base color has TAU weight, bloom color has 1-TAU)
            final_color = base_color.interpolate(bloom_color, 1 - TAU)
        else:
            final_color = base_color

        self.agent_colors[agent_id] = final_color
        return final_color

    def get_network_color(self) -> ColorState:
        """
        Get the overall network color based on collective state.

        The color mapping follows coherence levels:
            - Low coherence (< tau): Dim, desaturated
            - Mid coherence (tau to z_c): Activating colors
            - High coherence (> z_c): Vivid, full spectrum

        Returns:
            ColorState representing the network's collective chromatic state
        """
        return compute_network_coherence_color(self.garden)

    def export_chromatic_state(self) -> Dict[str, Any]:
        """
        Export the full chromatic state for visualization.

        Returns a comprehensive dictionary containing all color-related
        state information suitable for rendering in the Prismatic Engine
        or other visualization systems.

        Returns:
            Dictionary containing:
                - network_color: The overall network ColorState
                - agent_colors: Dict of agent_id -> ColorState data
                - gradient_blooms: List of bloom visualization data
                - color_blocks: List of block visualization data
                - corridors: Active spectral corridors
                - statistics: Chromatic statistics
        """
        network_color = self.get_network_color()

        # Compile agent colors
        agent_color_data = {}
        for agent_id in self.garden.agents:
            color = self.get_agent_color(agent_id)
            agent = self.garden.agents[agent_id]
            agent_color_data[agent_id] = {
                'name': agent.name,
                'color': color.to_dict(),
                'hex': color.to_hex(),
                'rgb': color.to_rgb(),
                'archetype': self._get_agent_archetype(agent_id),
            }

        # Get bloom visualization data
        bloom_data = [
            bloom.to_visualization_data() if hasattr(bloom, 'to_visualization_data')
            else bloom.to_dict()
            for bloom in self.gradient_blooms[-100:]  # Last 100 blooms
        ]

        # Get block visualization data
        block_data = [
            block.get_color_visualization()
            for block in self.color_blocks[-50:]  # Last 50 blocks
        ]

        # Determine active corridors
        active_corridors = self._get_active_corridors()

        # Compute statistics
        stats = self._compute_chromatic_stats()

        return {
            'timestamp': time.time(),
            'network_color': {
                'color': network_color.to_dict(),
                'hex': network_color.to_hex(),
                'coherence': network_color.coherence,
            },
            'agent_colors': agent_color_data,
            'gradient_blooms': bloom_data,
            'color_blocks': block_data,
            'corridors': active_corridors,
            'statistics': stats,
            'constants': {
                'phi': PHI,
                'tau': TAU,
                'z_c': Z_C,
                'k': K,
            },
        }

    def _get_agent_archetype(self, agent_id: str) -> Optional[str]:
        """Get the dominant archetype for an agent based on their blooms."""
        blooms = self._agent_bloom_history.get(agent_id, [])
        if not blooms:
            return None

        # Count archetypes
        archetype_counts: Dict[ArchetypeType, int] = {}
        for bloom in blooms:
            if bloom.dominant_archetype:
                archetype_counts[bloom.dominant_archetype] = (
                    archetype_counts.get(bloom.dominant_archetype, 0) + 1
                )

        if not archetype_counts:
            return None

        # Return most common
        dominant = max(archetype_counts.items(), key=lambda x: x[1])
        return dominant[0].value

    def _get_active_corridors(self) -> List[Dict[str, Any]]:
        """Get currently active spectral corridors."""
        if not self.gradient_blooms:
            return []

        # Analyze recent blooms for corridor activity
        recent_blooms = self.gradient_blooms[-20:]
        corridor_activity: Dict[CorridorType, int] = {}

        for bloom in recent_blooms:
            if bloom.corridor_id:
                corridor_activity[bloom.corridor_id] = (
                    corridor_activity.get(bloom.corridor_id, 0) + 1
                )

        # Return corridors with activity
        active = []
        for corridor_type, count in sorted(
            corridor_activity.items(),
            key=lambda x: x[1],
            reverse=True
        ):
            corridor = CORRIDORS.get(corridor_type)
            if corridor:
                colors = CORRIDOR_COLORS.get(corridor_type, ['#888888'])
                active.append({
                    'id': corridor_type.value,
                    'name': corridor_type.value.capitalize(),
                    'archetype': corridor.archetype,
                    'activity': count / len(recent_blooms),
                    'colors': colors,
                })

        return active

    def _compute_chromatic_stats(self) -> Dict[str, Any]:
        """Compute chromatic statistics for the garden."""
        if not self.gradient_blooms:
            return {
                'total_blooms': 0,
                'average_saturation': 0.5,
                'average_luminance': 0.5,
                'hue_diversity': 0.0,
                'archetype_distribution': {},
            }

        # Compute averages
        saturations = []
        luminances = []
        hues = []
        archetype_counts: Dict[str, int] = {}

        for bloom in self.gradient_blooms:
            if bloom.color_state:
                saturations.append(bloom.color_state.saturation)
                luminances.append(bloom.color_state.luminance)
                hues.append(bloom.color_state.hue)

            if bloom.dominant_archetype:
                key = bloom.dominant_archetype.value
                archetype_counts[key] = archetype_counts.get(key, 0) + 1

        # Hue diversity (circular variance)
        hue_diversity = 0.0
        if hues:
            h_sin = sum(math.sin(math.radians(h)) for h in hues) / len(hues)
            h_cos = sum(math.cos(math.radians(h)) for h in hues) / len(hues)
            hue_diversity = 1 - math.sqrt(h_sin**2 + h_cos**2)

        # Normalize archetype distribution
        total_archetypes = sum(archetype_counts.values())
        archetype_distribution = {
            k: v / total_archetypes
            for k, v in archetype_counts.items()
        } if total_archetypes > 0 else {}

        return {
            'total_blooms': len(self.gradient_blooms),
            'total_blocks': len(self.color_blocks),
            'average_saturation': sum(saturations) / len(saturations) if saturations else 0.5,
            'average_luminance': sum(luminances) / len(luminances) if luminances else 0.5,
            'hue_diversity': hue_diversity,
            'archetype_distribution': archetype_distribution,
        }

    def get_chromatic_vine(self) -> List[Dict[str, Any]]:
        """
        Get data for Chromatic Vine visualization.

        The Chromatic Vine is a timeline visualization showing the evolution
        of colors and archetypes through the garden's history. Each node
        represents a bloom event with its color, archetype, and connections.

        Returns:
            List of vine nodes with visualization data
        """
        vine_nodes = []

        for i, bloom in enumerate(self.gradient_blooms):
            # Get color hex
            color_hex = bloom.color_state.to_hex() if bloom.color_state else '#808080'

            # Get archetype info
            archetype_info = {}
            if bloom.dominant_archetype:
                arch_def = PERSONA_ARCHETYPES.get(bloom.dominant_archetype)
                if arch_def:
                    archetype_info = {
                        'type': bloom.dominant_archetype.value,
                        'name': arch_def.name,
                        'glyph': arch_def.glyph,
                    }

            # Get corridor info
            corridor_info = {}
            if bloom.corridor_id:
                corridor_info = {
                    'id': bloom.corridor_id.value,
                    'colors': CORRIDOR_COLORS.get(bloom.corridor_id, ['#888888']),
                }

            # Build node
            node = {
                'index': i,
                'event_id': bloom.event_id,
                'agent_id': bloom.agent_id,
                'timestamp': bloom.timestamp,
                'event_type': bloom.event_type,
                'significance': bloom.significance,
                'color_hex': color_hex,
                'color_rgb': bloom.color_state.to_rgb() if bloom.color_state else (128, 128, 128),
                'hsl': {
                    'h': bloom.color_state.hue if bloom.color_state else 0,
                    's': bloom.color_state.saturation if bloom.color_state else 0.5,
                    'l': bloom.color_state.luminance if bloom.color_state else 0.5,
                },
                'archetype': archetype_info,
                'corridor': corridor_info,
                'narrative': bloom.narrative_fragment,
                'lattice_coords': list(bloom.lattice_coords),
                # Connection to next node (for vine rendering)
                'next_index': i + 1 if i < len(self.gradient_blooms) - 1 else None,
            }

            vine_nodes.append(node)

        return vine_nodes

    def sync_from_garden(self):
        """
        Synchronize gradient blooms from the garden's bloom history.

        This method converts any existing bloom events in the garden
        that haven't been converted to gradient blooms yet.
        """
        # Build personality mapping
        personalities: Dict[str, AgentPersonalityTraits] = {}
        for agent_id, agent in self.garden.agents.items():
            personalities[agent_id] = personality_to_traits(agent.personality)

        # Track existing gradient bloom source IDs
        existing_source_ids = {
            gb.source_bloom_id for gb in self.gradient_blooms
            if gb.source_bloom_id
        }

        # Convert any new bloom events
        for bloom in self.garden.bloom_history:
            source_id = getattr(bloom, 'event_id', None)
            if source_id and source_id not in existing_source_ids:
                agent_id = getattr(bloom, 'primary_agent', None)
                personality = personalities.get(agent_id)

                gradient_bloom = from_bloom_event(bloom, personality)
                self.gradient_blooms.append(gradient_bloom)

                # Update agent history
                if agent_id:
                    if agent_id not in self._agent_bloom_history:
                        self._agent_bloom_history[agent_id] = []
                    self._agent_bloom_history[agent_id].append(gradient_bloom)

        # Update all agent colors
        for agent_id in self.garden.agents:
            self._update_agent_color(agent_id)


# =============================================================================
# Export Functions for Visualization
# =============================================================================

def export_for_prismatic_engine(garden: GradientGarden) -> Dict[str, Any]:
    """
    Export garden state for the Prismatic Engine visualization.

    The Prismatic Engine renders the garden as an interactive 3D
    crystalline structure where:
        - Each agent is a facet with their color
        - Bloom events create light pulses
        - Corridors are visible as spectral paths
        - Network coherence affects overall luminosity

    Args:
        garden: The GradientGarden to export

    Returns:
        Dictionary formatted for Prismatic Engine consumption
    """
    chromatic_state = garden.export_chromatic_state()

    # Transform to Prismatic Engine format
    return {
        'version': '1.0',
        'engine': 'prismatic',
        'timestamp': chromatic_state['timestamp'],

        # Global properties
        'ambient': {
            'color': chromatic_state['network_color']['hex'],
            'coherence': chromatic_state['network_color']['coherence'],
            'intensity': chromatic_state['network_color']['coherence'] ** PHI,
        },

        # Facets (agents)
        'facets': [
            {
                'id': agent_id,
                'name': data['name'],
                'color': data['hex'],
                'rgb': data['rgb'],
                'archetype': data['archetype'],
                'glow': garden.agent_colors.get(agent_id, ColorState()).coherence,
            }
            for agent_id, data in chromatic_state['agent_colors'].items()
        ],

        # Light pulses (recent blooms)
        'pulses': [
            {
                'id': bloom.get('event_id', ''),
                'source': bloom.get('agent_id', ''),
                'color': bloom.get('color_hex', '#808080'),
                'intensity': bloom.get('significance_raw', 0.5),
                'archetype_glyph': bloom.get('archetype', {}).get('glyph', ''),
                'timestamp': bloom.get('timestamp_iso', ''),
            }
            for bloom in chromatic_state['gradient_blooms'][-20:]
        ],

        # Spectral corridors
        'corridors': chromatic_state['corridors'],

        # Statistics for rendering hints
        'stats': chromatic_state['statistics'],

        # L4 constants for shader parameters
        'constants': chromatic_state['constants'],
    }


def export_for_chromatic_vine(garden: GradientGarden) -> Dict[str, Any]:
    """
    Export garden state for Chromatic Vine visualization.

    The Chromatic Vine is a 2D timeline visualization showing the
    evolution of colors through the garden's history, rendered as
    an organic vine structure with colored blooms.

    Args:
        garden: The GradientGarden to export

    Returns:
        Dictionary formatted for Chromatic Vine rendering
    """
    vine_nodes = garden.get_chromatic_vine()
    chromatic_state = garden.export_chromatic_state()

    return {
        'version': '1.0',
        'engine': 'chromatic_vine',
        'timestamp': time.time(),

        # Vine structure
        'vine': {
            'nodes': vine_nodes,
            'node_count': len(vine_nodes),
            'time_range': {
                'start': vine_nodes[0]['timestamp'] if vine_nodes else 0,
                'end': vine_nodes[-1]['timestamp'] if vine_nodes else 0,
            },
        },

        # Background/trunk color from network
        'trunk': {
            'color': chromatic_state['network_color']['hex'],
            'coherence': chromatic_state['network_color']['coherence'],
        },

        # Agent branches
        'branches': [
            {
                'agent_id': agent_id,
                'name': data['name'],
                'color': data['hex'],
                'node_count': len([
                    n for n in vine_nodes if n['agent_id'] == agent_id
                ]),
            }
            for agent_id, data in chromatic_state['agent_colors'].items()
        ],

        # Active corridors as vine paths
        'paths': chromatic_state['corridors'],

        # Statistics
        'stats': chromatic_state['statistics'],
    }
