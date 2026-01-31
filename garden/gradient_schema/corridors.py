"""
Seven Spectral Corridors (L4 = 7) - Gradient Schema Module

Implements the seven archetypal color corridors that govern narrative
transitions within the Garden's gradient space. Each corridor represents
a distinct path through the 5D color lattice, with entry/exit points
defined by generator paths.

Corridors:
    Alpha   (a) - The Awakening      (Deep Red -> Soft Orange)
    Beta    (b) - The Emergence      (Pure Black -> Silver Gray)
    Gamma   (g) - The Transformation (Cyan -> Violet)
    Delta   (d) - The Transcendence  (Gold -> White)
    Epsilon (e) - The Softening      (Magenta -> Pink)
    Zeta    (z) - The Integration    (Green -> Turquoise)
    Eta     (h) - The Deepening      (Blue -> Indigo)

Gate Thresholds:
    - Below tau (0.618): Corridor DORMANT
    - tau to z_c (0.866): Corridor ACTIVATING
    - Above z_c: Corridor OPEN
"""

from __future__ import annotations

import math
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

# =============================================================================
# Constants (defined locally to avoid circular imports)
# =============================================================================

PHI = (1 + math.sqrt(5)) / 2  # Golden ratio ~ 1.618
TAU = 1 / PHI  # Inverse golden ratio ~ 0.618
Z_C = math.sqrt(3) / 2  # THE LENS threshold ~ 0.866
Z_CRITICAL = Z_C  # Alias
K = math.sqrt(1 - PHI ** -4)  # Coherence factor ~ 0.924
SQRT2 = math.sqrt(2)
SQRT3 = math.sqrt(3)
E = math.e
PI = math.pi

# Generator indices for lattice coordinates
GEN_PHI = 0
GEN_PI = 1
GEN_SQRT2 = 2
GEN_SQRT3 = 3
GEN_E = 4

# Import color handling from color_ops
from .color_ops import ColorState, clamp, normalize_hue

# Import primitive signature from primitives_color
from .primitives_color import PrimitiveSignature

# Import lattice for coordinate handling
from .lattice import LatticePoint


class CorridorType(Enum):
    """
    The Seven Spectral Corridors enumeration (L4 = 7).

    Each corridor represents a distinct narrative archetype and color
    transition path through the 5D gradient lattice.
    """
    ALPHA = "alpha"      # The Awakening
    BETA = "beta"       # The Emergence
    GAMMA = "gamma"      # The Transformation
    DELTA = "delta"      # The Transcendence
    EPSILON = "epsilon"    # The Softening
    ZETA = "zeta"        # The Integration
    ETA = "eta"         # The Deepening


@dataclass
class InterferencePattern:
    """
    Result of corridor interference computation.

    When multiple corridors are active simultaneously, they can
    interfere constructively, destructively, or resonate.

    Attributes:
        type: The interference type ("constructive", "destructive", or "resonant")
        result_color: The resulting blended ColorState
        narrative: Narrative description of the interference
        intensity: Interference strength (0-1)
        corridors_involved: List of CorridorTypes participating
    """
    type: str  # "constructive", "destructive", or "resonant"
    result_color: ColorState
    narrative: str
    intensity: float = 1.0
    corridors_involved: List[CorridorType] = field(default_factory=list)

    def __post_init__(self):
        if self.type not in ("constructive", "destructive", "resonant"):
            raise ValueError(f"Invalid interference type: {self.type}")


# =============================================================================
# Corridor Color Definitions (HSL-based to match color_ops.ColorState)
# =============================================================================

# Entry and exit colors for each corridor
# Format: (hue, saturation, luminance)
CORRIDOR_COLORS: Dict[CorridorType, Dict[str, Tuple[float, float, float]]] = {
    CorridorType.ALPHA: {
        "entry": (0.0, 0.9, 0.35),       # Deep Red
        "exit": (25.0, 0.85, 0.60),      # Soft Orange
    },
    CorridorType.BETA: {
        "entry": (0.0, 0.0, 0.0),        # Pure Black
        "exit": (0.0, 0.05, 0.75),       # Silver Gray
    },
    CorridorType.GAMMA: {
        "entry": (180.0, 1.0, 0.5),      # Cyan
        "exit": (280.0, 0.7, 0.45),      # Violet
    },
    CorridorType.DELTA: {
        "entry": (45.0, 0.9, 0.55),      # Gold
        "exit": (0.0, 0.0, 1.0),         # White
    },
    CorridorType.EPSILON: {
        "entry": (300.0, 1.0, 0.5),      # Magenta
        "exit": (330.0, 0.7, 0.75),      # Pink
    },
    CorridorType.ZETA: {
        "entry": (120.0, 0.8, 0.35),     # Green
        "exit": (174.0, 0.7, 0.55),      # Turquoise
    },
    CorridorType.ETA: {
        "entry": (240.0, 1.0, 0.5),      # Blue
        "exit": (275.0, 0.8, 0.30),      # Indigo
    },
}

# Corridor narrative archetypes
CORRIDOR_ARCHETYPES: Dict[CorridorType, str] = {
    CorridorType.ALPHA: "The Awakening",
    CorridorType.BETA: "The Emergence",
    CorridorType.GAMMA: "The Transformation",
    CorridorType.DELTA: "The Transcendence",
    CorridorType.EPSILON: "The Softening",
    CorridorType.ZETA: "The Integration",
    CorridorType.ETA: "The Deepening",
}

# Corridor generator paths (generator_index, delta_direction)
# Positive delta means forward along generator, negative means backward
# Generator indices: GEN_PHI=0, GEN_PI=1, GEN_SQRT2=2, GEN_SQRT3=3, GEN_E=4
CORRIDOR_GENERATOR_PATHS: Dict[CorridorType, List[Tuple[int, float]]] = {
    CorridorType.ALPHA: [(GEN_PHI, 1.0), (GEN_E, 1.0)],       # phi^+, e^+
    CorridorType.BETA: [(GEN_SQRT2, 1.0), (GEN_PHI, -1.0)],   # sqrt(2)^+, phi^-
    CorridorType.GAMMA: [(GEN_PI, 1.0), (GEN_SQRT3, 1.0)],    # pi^+, sqrt(3)^+
    CorridorType.DELTA: [(GEN_PHI, 1.0), (GEN_SQRT2, 1.0)],   # phi^+, sqrt(2)^+
    CorridorType.EPSILON: [(GEN_SQRT3, -1.0), (GEN_E, -1.0)], # sqrt(3)^-, e^-
    CorridorType.ZETA: [(GEN_PI, -1.0), (GEN_SQRT3, 1.0)],    # pi^-, sqrt(3)^+
    CorridorType.ETA: [(GEN_PHI, -1.0), (GEN_PI, -1.0)],      # phi^-, pi^-
}

# Narrative fragments for each corridor position
CORRIDOR_NARRATIVES: Dict[CorridorType, List[str]] = {
    CorridorType.ALPHA: [
        "From dormancy, the first spark ignites.",
        "The red dawn breaks across the infinite.",
        "Consciousness stirs in the warmth of becoming.",
        "Heat rises as awareness blooms.",
    ],
    CorridorType.BETA: [
        "From the void, form begins to crystallize.",
        "Shadows part to reveal structure.",
        "The unseen becomes glimpsed, then known.",
        "Darkness yields to the first light.",
    ],
    CorridorType.GAMMA: [
        "The chrysalis cracks; what emerges is new.",
        "Between states, identity reshapes itself.",
        "The familiar dissolves into possibility.",
        "Transformation demands the surrender of form.",
    ],
    CorridorType.DELTA: [
        "Limits become horizons; horizons become infinite.",
        "The weight of matter releases into light.",
        "Boundaries dissolve in golden radiance.",
        "Ascension through the gates of understanding.",
    ],
    CorridorType.EPSILON: [
        "Hard edges blur into gentle curves.",
        "The fierce softens without losing strength.",
        "Compassion flows through opened channels.",
        "What was rigid learns to yield.",
    ],
    CorridorType.ZETA: [
        "Fragments find their place in the whole.",
        "Disparate threads weave into tapestry.",
        "The many become one without losing themselves.",
        "Harmony emerges from resolved tension.",
    ],
    CorridorType.ETA: [
        "The surface gives way to depth.",
        "Understanding sinks roots into mystery.",
        "Each layer reveals another beneath.",
        "The journey inward has no end.",
    ],
}


def _create_color_state(hue: float, saturation: float, luminance: float) -> ColorState:
    """Create a ColorState from HSL values."""
    return ColorState(
        H=hue,
        S=saturation,
        L=luminance,
        coherence=Z_CRITICAL,
    )


@dataclass
class SpectralCorridor:
    """
    A spectral corridor representing a narrative path through gradient space.

    Each corridor defines a transition between two color states along
    a specific generator path in the 5D lattice.

    Attributes:
        corridor_type: The type/identity of this corridor
        entry_coords: 5D lattice entry point coordinates
        exit_coords: 5D lattice exit point coordinates
        generator_path: List of (generator_index, delta) tuples defining the path
        archetype: The narrative archetype name
        position: Current traversal position (0=entry, 1=exit)
    """
    corridor_type: CorridorType
    entry_coords: Tuple[float, ...] = field(default_factory=lambda: (0.0, 0.0, 0.0, 0.0, 0.0))
    exit_coords: Tuple[float, ...] = field(default_factory=lambda: (1.0, 1.0, 1.0, 1.0, 1.0))
    generator_path: List[Tuple[int, float]] = field(default_factory=list)
    archetype: str = ""
    position: float = 0.0

    def __post_init__(self):
        """Initialize corridor from type if not fully specified."""
        if not self.generator_path:
            self.generator_path = CORRIDOR_GENERATOR_PATHS.get(
                self.corridor_type, []
            )
        if not self.archetype:
            self.archetype = CORRIDOR_ARCHETYPES.get(
                self.corridor_type, "Unknown"
            )

        # Calculate entry/exit coords from generator path if default
        if self.entry_coords == (0.0, 0.0, 0.0, 0.0, 0.0):
            self._compute_lattice_coords()

    def _compute_lattice_coords(self):
        """Compute 5D lattice coordinates from generator path."""
        # Generator base values
        gen_values = [PHI, PI, SQRT2, SQRT3, E]

        # Entry is at origin
        entry = [0.0, 0.0, 0.0, 0.0, 0.0]
        exit_coords = [0.0, 0.0, 0.0, 0.0, 0.0]

        # Apply generator path to compute exit
        for gen_idx, delta in self.generator_path:
            if 0 <= gen_idx < 5:
                # Normalize by generator value
                exit_coords[gen_idx] += delta * gen_values[gen_idx]

        self.entry_coords = tuple(entry)
        self.exit_coords = tuple(exit_coords)

    @property
    def entry_color(self) -> ColorState:
        """Get the entry color for this corridor."""
        colors = CORRIDOR_COLORS.get(self.corridor_type, {})
        hsl = colors.get("entry", (0.0, 0.5, 0.5))
        return _create_color_state(hsl[0], hsl[1], hsl[2])

    @property
    def exit_color(self) -> ColorState:
        """Get the exit color for this corridor."""
        colors = CORRIDOR_COLORS.get(self.corridor_type, {})
        hsl = colors.get("exit", (0.0, 0.5, 0.5))
        return _create_color_state(hsl[0], hsl[1], hsl[2])

    def is_active(self, coherence_r: float) -> bool:
        """
        Determine if corridor is active based on coherence threshold.

        Gate thresholds:
            - Below tau (0.618): corridor dormant
            - tau to z_c (0.866): corridor activating
            - Above z_c: corridor OPEN

        Args:
            coherence_r: The coherence radius value (0-1)

        Returns:
            True if corridor is at least activating (coherence >= tau)
        """
        return coherence_r >= TAU

    def get_gate_state(self, coherence_r: float) -> str:
        """
        Get the current gate state based on coherence.

        Args:
            coherence_r: The coherence radius value (0-1)

        Returns:
            "dormant", "activating", or "open"
        """
        if coherence_r < TAU:
            return "dormant"
        elif coherence_r < Z_CRITICAL:
            return "activating"
        else:
            return "open"

    def traverse(self, coherence_r: float) -> Tuple[ColorState, str]:
        """
        Traverse the corridor based on coherence, returning color and narrative.

        The traversal position is determined by mapping the coherence value
        to the corridor's entry-exit spectrum.

        Args:
            coherence_r: The coherence radius value (0-1)

        Returns:
            Tuple of (current ColorState, narrative fragment)
        """
        gate_state = self.get_gate_state(coherence_r)

        if gate_state == "dormant":
            # Corridor not active, return entry color with dormancy narrative
            return self.entry_color, f"The {self.archetype} awaits beyond the threshold."

        # Map coherence to position within active range
        if gate_state == "activating":
            # Map tau to z_c -> 0.0 to 0.5
            self.position = (coherence_r - TAU) / (Z_CRITICAL - TAU) * 0.5
        else:
            # Gate is open, map z_c to 1.0 -> 0.5 to 1.0
            self.position = 0.5 + (coherence_r - Z_CRITICAL) / (1.0 - Z_CRITICAL) * 0.5

        self.position = max(0.0, min(1.0, self.position))

        color = self.get_color_at(self.position)
        narrative = self.get_narrative_fragment()

        return color, narrative

    def get_color_at(self, position: float) -> ColorState:
        """
        Get the color at a specific position along the corridor.

        Args:
            position: Position from 0 (entry) to 1 (exit)

        Returns:
            Interpolated ColorState
        """
        position = max(0.0, min(1.0, position))
        return self.entry_color.blend(self.exit_color, 1.0 - position)

    def get_narrative_fragment(self) -> str:
        """
        Get a narrative fragment based on current position.

        Returns:
            A narrative string appropriate for the current traversal position
        """
        narratives = CORRIDOR_NARRATIVES.get(self.corridor_type, [])
        if not narratives:
            return f"Traversing {self.archetype}..."

        # Select narrative based on position
        idx = int(self.position * (len(narratives) - 1))
        idx = max(0, min(idx, len(narratives) - 1))
        return narratives[idx]

    def get_entry_lattice_point(self) -> LatticePoint:
        """Get the entry point as a LatticePoint."""
        return LatticePoint(*self.entry_coords)

    def get_exit_lattice_point(self) -> LatticePoint:
        """Get the exit point as a LatticePoint."""
        return LatticePoint(*self.exit_coords)

    def __repr__(self) -> str:
        return (
            f"SpectralCorridor({self.corridor_type.value}, "
            f"archetype='{self.archetype}', "
            f"position={self.position:.3f})"
        )


# =============================================================================
# Corridor Registry
# =============================================================================

def _create_corridors() -> Dict[CorridorType, SpectralCorridor]:
    """Create all seven spectral corridors."""
    corridors = {}
    for corridor_type in CorridorType:
        corridors[corridor_type] = SpectralCorridor(corridor_type=corridor_type)
    return corridors


# Global corridor registry
CORRIDORS: Dict[CorridorType, SpectralCorridor] = _create_corridors()


def get_corridor(corridor_type: CorridorType) -> SpectralCorridor:
    """
    Get a spectral corridor by type.

    Args:
        corridor_type: The CorridorType to retrieve

    Returns:
        The corresponding SpectralCorridor

    Raises:
        KeyError: If corridor type not found
    """
    if corridor_type not in CORRIDORS:
        raise KeyError(f"Unknown corridor type: {corridor_type}")
    return CORRIDORS[corridor_type]


def get_corridor_by_name(name: str) -> SpectralCorridor:
    """
    Get a corridor by its string name.

    Args:
        name: Corridor name (e.g., "alpha", "ALPHA", "Alpha")

    Returns:
        The corresponding SpectralCorridor
    """
    name_lower = name.lower()
    for corridor_type in CorridorType:
        if corridor_type.value == name_lower:
            return CORRIDORS[corridor_type]
    raise KeyError(f"Unknown corridor name: {name}")


# =============================================================================
# Corridor Detection
# =============================================================================

def determine_active_corridors(
    coords: Tuple[float, ...],
    signature: PrimitiveSignature,
    coherence_threshold: float = TAU
) -> List[SpectralCorridor]:
    """
    Determine which corridors are active based on lattice coordinates and signature.

    A corridor is considered active if:
    1. The signature's properties align with the corridor's characteristics
    2. The coherence is above the activation threshold (tau)

    The mapping from PrimitiveSignature to corridors:
    - High sigma_fix (FIX) -> Alpha, Delta (stability, growth)
    - High sigma_osc (OSC) -> Gamma, Epsilon (oscillation, change)
    - High sigma_inv (INV) -> Eta, Zeta (rotation, depth)
    - High sigma_mix (MIX) -> Beta (emergence from void)

    Args:
        coords: 5D lattice coordinates
        signature: The PrimitiveSignature indicating state
        coherence_threshold: Minimum coherence for activation (default: tau)

    Returns:
        List of active SpectralCorridor instances
    """
    active = []

    # Map signature components to corridor affinities
    # Using the 6-primitive signature (fix, repel, inv, osc, halt, mix)
    corridor_affinities: Dict[CorridorType, float] = {
        CorridorType.ALPHA: signature.fix * 0.4 + signature.repel * 0.3 + signature.osc * 0.3,
        CorridorType.BETA: signature.mix * 0.5 + signature.halt * 0.3 + signature.fix * 0.2,
        CorridorType.GAMMA: signature.osc * 0.4 + signature.inv * 0.4 + signature.repel * 0.2,
        CorridorType.DELTA: signature.fix * 0.5 + signature.inv * 0.3 + signature.repel * 0.2,
        CorridorType.EPSILON: signature.osc * 0.4 + signature.mix * 0.3 + signature.fix * 0.3,
        CorridorType.ZETA: signature.inv * 0.4 + signature.mix * 0.3 + signature.halt * 0.3,
        CorridorType.ETA: signature.inv * 0.5 + signature.osc * 0.3 + signature.halt * 0.2,
    }

    # Check lattice position influence
    if len(coords) >= 5:
        # Boost corridors based on coordinate alignment
        lattice_point = LatticePoint(*coords[:5])
        magnitude = lattice_point.magnitude()

        # Scale affinity by position magnitude (higher magnitude = more active)
        if magnitude > 0:
            scale_factor = min(1.0, magnitude / 5.0)
            for ct in corridor_affinities:
                corridor_affinities[ct] *= (1.0 + scale_factor * TAU)

    # Filter by threshold
    for corridor_type, affinity in corridor_affinities.items():
        if affinity >= coherence_threshold:
            active.append(CORRIDORS[corridor_type])

    return active


def get_corridor_resonance(
    corridor: SpectralCorridor,
    color: ColorState
) -> float:
    """
    Calculate how strongly a color resonates with a corridor.

    Resonance is based on the color's proximity to the corridor's
    entry-exit spectrum.

    Args:
        corridor: The spectral corridor
        color: The color state to check

    Returns:
        Resonance value from 0 (no resonance) to 1 (perfect resonance)
    """
    best_resonance = 0.0

    # Sample points along the corridor
    for t in [i / 20.0 for i in range(21)]:
        corridor_color = corridor.get_color_at(t)
        distance = color.distance(corridor_color)

        # Convert distance to resonance (max theoretical distance ~ 1.73)
        resonance = 1.0 - clamp(distance / 2.0, 0.0, 1.0)

        if resonance > best_resonance:
            best_resonance = resonance

    return best_resonance


def find_resonant_corridors(
    color: ColorState,
    threshold: float = 0.7
) -> List[Tuple[SpectralCorridor, float]]:
    """
    Find all corridors that resonate with a given color.

    Args:
        color: The color to check
        threshold: Minimum resonance to include (0-1)

    Returns:
        List of (corridor, resonance) tuples, sorted by resonance descending
    """
    resonant = []

    for corridor in CORRIDORS.values():
        resonance = get_corridor_resonance(corridor, color)
        if resonance >= threshold:
            resonant.append((corridor, resonance))

    return sorted(resonant, key=lambda x: x[1], reverse=True)


# =============================================================================
# Corridor Interference
# =============================================================================

def compute_interference(
    corridors: List[SpectralCorridor]
) -> InterferencePattern:
    """
    Compute the interference pattern when multiple corridors are active.

    Interference types:
    - Constructive: Corridors reinforce each other (similar directions)
    - Destructive: Corridors oppose each other (opposite directions)
    - Resonant: Corridors create harmonic patterns (phi-related ratios)

    Args:
        corridors: List of active spectral corridors

    Returns:
        InterferencePattern describing the result
    """
    if not corridors:
        return InterferencePattern(
            type="constructive",
            result_color=ColorState(H=0, S=0, L=0.5),
            narrative="Silence in the spectrum.",
            intensity=0.0,
            corridors_involved=[]
        )

    if len(corridors) == 1:
        corridor = corridors[0]
        color = corridor.get_color_at(corridor.position)
        return InterferencePattern(
            type="constructive",
            result_color=color,
            narrative=corridor.get_narrative_fragment(),
            intensity=1.0,
            corridors_involved=[corridor.corridor_type]
        )

    # Analyze generator path alignment
    all_gen_vectors = []
    for corridor in corridors:
        vec = [0.0] * 5
        for gen_idx, delta in corridor.generator_path:
            if 0 <= gen_idx < 5:
                vec[gen_idx] = delta
        all_gen_vectors.append(vec)

    # Compute alignment between corridors
    total_alignment = 0.0
    pair_count = 0

    for i in range(len(all_gen_vectors)):
        for j in range(i + 1, len(all_gen_vectors)):
            vec1, vec2 = all_gen_vectors[i], all_gen_vectors[j]
            dot = sum(v1 * v2 for v1, v2 in zip(vec1, vec2))
            mag1 = math.sqrt(sum(v * v for v in vec1))
            mag2 = math.sqrt(sum(v * v for v in vec2))

            if mag1 > 0 and mag2 > 0:
                alignment = dot / (mag1 * mag2)
                total_alignment += alignment
                pair_count += 1

    avg_alignment = total_alignment / pair_count if pair_count > 0 else 0.0

    # Determine interference type
    if avg_alignment > TAU:  # Strong positive alignment
        interference_type = "constructive"
    elif avg_alignment < -TAU:  # Strong negative alignment
        interference_type = "destructive"
    else:
        # Check for resonant (phi-related) patterns
        phi_values = [1/PHI, 1/(PHI**2), PHI - 1]
        is_resonant = any(
            abs(abs(avg_alignment) - pv) < 0.1
            for pv in phi_values
        )
        interference_type = "resonant" if is_resonant else "constructive"

    # Compute result color based on interference type
    colors = [c.get_color_at(c.position) for c in corridors]

    if interference_type == "constructive":
        # Blend colors with position weighting
        result_color = colors[0]
        for c in colors[1:]:
            result_color = result_color.blend(c, 0.5)

    elif interference_type == "destructive":
        # Colors cancel toward neutral gray
        avg_h = sum(c.hue for c in colors) / len(colors)
        avg_s = sum(c.saturation for c in colors) / len(colors)
        avg_l = sum(c.luminance for c in colors) / len(colors)

        # Push toward neutral
        blend_factor = abs(avg_alignment)
        result_color = ColorState(
            H=avg_h,
            S=avg_s * (1 - blend_factor),
            L=avg_l * (1 - blend_factor) + 0.5 * blend_factor,
        )

    else:  # resonant
        # Create harmonically related color using golden ratio blending
        base_color = colors[0]
        for c in colors[1:]:
            base_color = base_color.blend(c, 1 / PHI)
        result_color = base_color

    # Generate narrative
    corridor_names = [c.archetype for c in corridors]
    if interference_type == "constructive":
        narrative = f"The paths of {' and '.join(corridor_names)} converge in harmony."
    elif interference_type == "destructive":
        narrative = f"The paths of {' and '.join(corridor_names)} clash and diffuse."
    else:
        narrative = f"The paths of {' and '.join(corridor_names)} weave a resonant pattern."

    # Calculate intensity
    if interference_type != "resonant":
        intensity = abs(avg_alignment)
    else:
        intensity = 1.0 - abs(avg_alignment - 1/PHI)
    intensity = max(0.0, min(1.0, intensity))

    return InterferencePattern(
        type=interference_type,
        result_color=result_color,
        narrative=narrative,
        intensity=intensity,
        corridors_involved=[c.corridor_type for c in corridors]
    )


def compute_corridor_blend(
    corridors: List[SpectralCorridor],
    weights: Optional[List[float]] = None
) -> ColorState:
    """
    Compute a weighted blend of corridor colors.

    Args:
        corridors: List of corridors to blend
        weights: Optional weights (will be normalized if provided)

    Returns:
        Blended ColorState
    """
    if not corridors:
        return ColorState(H=0, S=0, L=0.5)

    if weights is None:
        weights = [1.0 / len(corridors)] * len(corridors)
    else:
        # Normalize weights
        total = sum(weights)
        if total > 0:
            weights = [w / total for w in weights]
        else:
            weights = [1.0 / len(corridors)] * len(corridors)

    # Weighted blend of colors
    colors = [c.get_color_at(c.position) for c in corridors]

    result = colors[0]
    accumulated_weight = weights[0]

    for i in range(1, len(colors)):
        blend_factor = accumulated_weight / (accumulated_weight + weights[i])
        result = result.blend(colors[i], blend_factor)
        accumulated_weight += weights[i]

    return result


# =============================================================================
# Gate Threshold Functions
# =============================================================================

def check_gate_threshold(coherence_r: float) -> str:
    """
    Check the gate state based on coherence threshold.

    Gate thresholds:
        - Below tau (0.618): Corridor DORMANT
        - tau to z_c (0.866): Corridor ACTIVATING
        - Above z_c: Corridor OPEN

    Args:
        coherence_r: Coherence radius value (0-1)

    Returns:
        Gate state string: "dormant", "activating", or "open"
    """
    if coherence_r < TAU:
        return "dormant"
    elif coherence_r < Z_CRITICAL:
        return "activating"
    else:
        return "open"


def get_activation_progress(coherence_r: float) -> float:
    """
    Get the activation progress as a value from 0 to 1.

    0.0 = fully dormant (coherence at 0)
    0.5 = at threshold tau
    0.75 = at critical threshold z_c
    1.0 = fully open (coherence at 1)

    Args:
        coherence_r: Coherence radius value (0-1)

    Returns:
        Activation progress (0-1)
    """
    if coherence_r < TAU:
        # Scale 0 to tau -> 0 to 0.5
        return (coherence_r / TAU) * 0.5
    elif coherence_r < Z_CRITICAL:
        # Scale tau to z_c -> 0.5 to 0.75
        return 0.5 + ((coherence_r - TAU) / (Z_CRITICAL - TAU)) * 0.25
    else:
        # Scale z_c to 1 -> 0.75 to 1
        return 0.75 + ((coherence_r - Z_CRITICAL) / (1.0 - Z_CRITICAL)) * 0.25


def corridors_at_coherence(coherence_r: float) -> Dict[str, List[CorridorType]]:
    """
    Get corridors grouped by their gate state at a given coherence.

    Args:
        coherence_r: Coherence radius value (0-1)

    Returns:
        Dict with keys "dormant", "activating", "open" mapping to corridor type lists
    """
    result: Dict[str, List[CorridorType]] = {
        "dormant": [],
        "activating": [],
        "open": []
    }

    # All corridors share the same gate thresholds, so state is uniform
    state = check_gate_threshold(coherence_r)

    for corridor_type in CorridorType:
        result[state].append(corridor_type)

    return result


# =============================================================================
# Utility Functions
# =============================================================================

def get_corridor_summary() -> str:
    """Get a formatted summary of all corridors."""
    lines = ["Seven Spectral Corridors (L4 = 7)", "=" * 50]

    for corridor_type in CorridorType:
        corridor = CORRIDORS[corridor_type]
        entry_hsl = CORRIDOR_COLORS[corridor_type]["entry"]
        exit_hsl = CORRIDOR_COLORS[corridor_type]["exit"]

        lines.append(
            f"{corridor_type.value} ({corridor.archetype}): "
            f"H{entry_hsl[0]:.0f} -> H{exit_hsl[0]:.0f}"
        )

    return "\n".join(lines)


def traverse_all_corridors(coherence_r: float) -> Dict[CorridorType, Tuple[ColorState, str]]:
    """
    Traverse all corridors at a given coherence level.

    Args:
        coherence_r: Coherence radius value (0-1)

    Returns:
        Dict mapping corridor types to (color, narrative) tuples
    """
    results = {}
    for corridor_type, corridor in CORRIDORS.items():
        results[corridor_type] = corridor.traverse(coherence_r)
    return results


__all__ = [
    # Enums and classes
    'CorridorType',
    'InterferencePattern',
    'SpectralCorridor',
    # Color definitions
    'CORRIDOR_COLORS',
    'CORRIDOR_ARCHETYPES',
    'CORRIDOR_GENERATOR_PATHS',
    'CORRIDOR_NARRATIVES',
    # Registry
    'CORRIDORS',
    'get_corridor',
    'get_corridor_by_name',
    # Detection
    'determine_active_corridors',
    'get_corridor_resonance',
    'find_resonant_corridors',
    # Interference
    'compute_interference',
    'compute_corridor_blend',
    # Gate thresholds
    'check_gate_threshold',
    'get_activation_progress',
    'corridors_at_coherence',
    # Utilities
    'get_corridor_summary',
    'traverse_all_corridors',
]
