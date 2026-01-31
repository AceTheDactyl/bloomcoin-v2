"""
Narrative Archetypes - F12 Archetypal Patterns for the Gradient Schema
======================================================================

The F12 = 144 narrative variants arise from combinations of the seven
spectral corridors with archetypal story patterns. These archetypes
provide semantic meaning to color transitions in the Garden blockchain.

Each archetype maps a narrative pattern to a color trajectory, enabling
AI memory states to be visualized as story-like progressions.

Core Archetypes (12 Primary):
    1. EMERGENCE    - From darkness to light (low L -> high L)
    2. DESCENT      - From light to shadow (high L -> low L)
    3. QUEST        - Warm hue journey (Gold -> Crimson -> Amber)
    4. RETURN       - Cool hue journey (Azure -> Verdant -> Gold)
    5. TRANSFORMATION - Saturation shift (low S -> high S)
    6. DISSOLUTION  - Saturation loss (high S -> low S)
    7. CONFLICT     - Opponent hues (complementary tension)
    8. HARMONY      - Adjacent hues (analogous flow)
    9. THRESHOLD    - At Z_C boundary (liminal states)
    10. CATALYST    - High gradient rate (rapid change)
    11. STASIS      - Low gradient rate (stability)
    12. CYCLE       - Full hue rotation (return to origin)

The 144 variants arise from: 12 archetypes x 7 corridors x ~2 directions
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional, Callable, Dict
from enum import Enum, auto
import math

from . import PHI, TAU, Z_C, K, F12, L4


class ArchetypeClass(Enum):
    """Classification of archetype narrative patterns."""
    LUMINANCE = auto()     # Light/dark transitions
    HUE = auto()           # Color journey
    SATURATION = auto()    # Intensity changes
    TENSION = auto()       # Conflict/harmony
    THRESHOLD = auto()     # Liminal states
    TEMPORAL = auto()      # Rate-based patterns


@dataclass
class NarrativeArchetype:
    """
    A narrative archetype defining a color-story pattern.

    Attributes:
        id: Unique archetype identifier (0-143)
        name: Human-readable archetype name
        archetype_class: Classification category
        description: Narrative description
        start_conditions: Dict of starting HSL constraints
        end_conditions: Dict of ending HSL constraints
        trajectory: Function defining the path (optional)
        corridor_sequence: Preferred corridor sequence (optional)
    """
    id: int
    name: str
    archetype_class: ArchetypeClass
    description: str
    start_conditions: Dict[str, Tuple[float, float]]  # {param: (min, max)}
    end_conditions: Dict[str, Tuple[float, float]]
    corridor_sequence: Optional[Tuple[int, ...]] = None

    def matches_start(self, h: float, s: float, l: float) -> bool:
        """Check if HSL values match start conditions."""
        values = {'h': h, 's': s, 'l': l}
        for param, (min_val, max_val) in self.start_conditions.items():
            if param == 'h':
                # Handle circular hue
                val = values[param] % 360
                if min_val <= max_val:
                    if not (min_val <= val <= max_val):
                        return False
                else:  # Wraps around 360
                    if not (val >= min_val or val <= max_val):
                        return False
            else:
                if not (min_val <= values[param] <= max_val):
                    return False
        return True

    def matches_end(self, h: float, s: float, l: float) -> bool:
        """Check if HSL values match end conditions."""
        values = {'h': h, 's': s, 'l': l}
        for param, (min_val, max_val) in self.end_conditions.items():
            if param == 'h':
                val = values[param] % 360
                if min_val <= max_val:
                    if not (min_val <= val <= max_val):
                        return False
                else:
                    if not (val >= min_val or val <= max_val):
                        return False
            else:
                if not (min_val <= values[param] <= max_val):
                    return False
        return True

    def score_transition(self, start_hsl: Tuple[float, float, float],
                         end_hsl: Tuple[float, float, float]) -> float:
        """
        Score how well a transition matches this archetype.

        Returns a value from 0 (no match) to 1 (perfect match).
        """
        start_match = 1.0 if self.matches_start(*start_hsl) else 0.0
        end_match = 1.0 if self.matches_end(*end_hsl) else 0.0
        return (start_match + end_match) / 2


# Define the 12 primary archetypes

ARCHETYPE_EMERGENCE = NarrativeArchetype(
    id=0,
    name="Emergence",
    archetype_class=ArchetypeClass.LUMINANCE,
    description="Rising from darkness into light - awakening, discovery, hope",
    start_conditions={'l': (0.0, 0.4)},
    end_conditions={'l': (0.6, 1.0)},
    corridor_sequence=(6, 0, 3),  # Silver -> Gold -> Verdant
)

ARCHETYPE_DESCENT = NarrativeArchetype(
    id=1,
    name="Descent",
    archetype_class=ArchetypeClass.LUMINANCE,
    description="Falling into shadow - introspection, mystery, depth",
    start_conditions={'l': (0.6, 1.0)},
    end_conditions={'l': (0.0, 0.4)},
    corridor_sequence=(0, 4, 6),  # Gold -> Violet -> Silver
)

ARCHETYPE_QUEST = NarrativeArchetype(
    id=2,
    name="Quest",
    archetype_class=ArchetypeClass.HUE,
    description="Journey through warm hues - action, passion, achievement",
    start_conditions={'h': (30, 60), 's': (0.4, 1.0)},
    end_conditions={'h': (0, 30), 's': (0.4, 1.0)},
    corridor_sequence=(0, 5, 1),  # Gold -> Amber -> Crimson
)

ARCHETYPE_RETURN = NarrativeArchetype(
    id=3,
    name="Return",
    archetype_class=ArchetypeClass.HUE,
    description="Journey through cool hues - reflection, wisdom, homecoming",
    start_conditions={'h': (180, 270)},
    end_conditions={'h': (30, 60)},
    corridor_sequence=(2, 3, 0),  # Azure -> Verdant -> Gold
)

ARCHETYPE_TRANSFORMATION = NarrativeArchetype(
    id=4,
    name="Transformation",
    archetype_class=ArchetypeClass.SATURATION,
    description="Intensification - growth, passion, commitment",
    start_conditions={'s': (0.0, 0.4)},
    end_conditions={'s': (0.6, 1.0)},
)

ARCHETYPE_DISSOLUTION = NarrativeArchetype(
    id=5,
    name="Dissolution",
    archetype_class=ArchetypeClass.SATURATION,
    description="Fading intensity - release, peace, transcendence",
    start_conditions={'s': (0.6, 1.0)},
    end_conditions={'s': (0.0, 0.4)},
    corridor_sequence=(1, 0, 6),  # Crimson -> Gold -> Silver
)

ARCHETYPE_CONFLICT = NarrativeArchetype(
    id=6,
    name="Conflict",
    archetype_class=ArchetypeClass.TENSION,
    description="Tension between opposites - struggle, drama, resolution",
    start_conditions={'s': (0.5, 1.0)},
    end_conditions={'s': (0.5, 1.0)},
    corridor_sequence=(1, 4),  # Crimson -> Violet (complementary)
)

ARCHETYPE_HARMONY = NarrativeArchetype(
    id=7,
    name="Harmony",
    archetype_class=ArchetypeClass.TENSION,
    description="Flow between neighbors - peace, unity, cooperation",
    start_conditions={},  # Any start
    end_conditions={},    # Judged by hue proximity
    corridor_sequence=(0, 5, 3),  # Gold -> Amber -> Verdant (analogous)
)

ARCHETYPE_THRESHOLD = NarrativeArchetype(
    id=8,
    name="Threshold",
    archetype_class=ArchetypeClass.THRESHOLD,
    description="At the boundary - liminality, transition, potential",
    start_conditions={'l': (Z_C - 0.1, Z_C + 0.1)},
    end_conditions={'l': (Z_C - 0.1, Z_C + 0.1)},
    corridor_sequence=(0,),  # Stays at Gold (origin)
)

ARCHETYPE_CATALYST = NarrativeArchetype(
    id=9,
    name="Catalyst",
    archetype_class=ArchetypeClass.TEMPORAL,
    description="Rapid change - crisis, breakthrough, acceleration",
    start_conditions={'s': (0.3, 1.0)},
    end_conditions={'s': (0.3, 1.0)},
)

ARCHETYPE_STASIS = NarrativeArchetype(
    id=10,
    name="Stasis",
    archetype_class=ArchetypeClass.TEMPORAL,
    description="Stillness - meditation, stability, endurance",
    start_conditions={},
    end_conditions={},
)

ARCHETYPE_CYCLE = NarrativeArchetype(
    id=11,
    name="Cycle",
    archetype_class=ArchetypeClass.HUE,
    description="Full rotation and return - seasons, rebirth, eternal return",
    start_conditions={'h': (30, 60)},
    end_conditions={'h': (30, 60)},
    corridor_sequence=(0, 3, 2, 4, 1, 5, 0),  # Full corridor cycle
)

# List of primary archetypes
PRIMARY_ARCHETYPES: List[NarrativeArchetype] = [
    ARCHETYPE_EMERGENCE,
    ARCHETYPE_DESCENT,
    ARCHETYPE_QUEST,
    ARCHETYPE_RETURN,
    ARCHETYPE_TRANSFORMATION,
    ARCHETYPE_DISSOLUTION,
    ARCHETYPE_CONFLICT,
    ARCHETYPE_HARMONY,
    ARCHETYPE_THRESHOLD,
    ARCHETYPE_CATALYST,
    ARCHETYPE_STASIS,
    ARCHETYPE_CYCLE,
]

# Dictionary for lookup by name
ARCHETYPE_BY_NAME: Dict[str, NarrativeArchetype] = {
    a.name.lower(): a for a in PRIMARY_ARCHETYPES
}


def get_archetype(id_or_name) -> NarrativeArchetype:
    """
    Get an archetype by ID or name.

    Args:
        id_or_name: Integer ID (0-11 for primary) or name string

    Returns:
        The corresponding NarrativeArchetype

    Raises:
        ValueError: If archetype not found
    """
    if isinstance(id_or_name, int):
        if 0 <= id_or_name < len(PRIMARY_ARCHETYPES):
            return PRIMARY_ARCHETYPES[id_or_name]
        raise ValueError(f"Archetype ID must be 0-{len(PRIMARY_ARCHETYPES)-1}")

    name = str(id_or_name).lower()
    if name in ARCHETYPE_BY_NAME:
        return ARCHETYPE_BY_NAME[name]
    raise ValueError(f"Unknown archetype name: {id_or_name}")


def classify_transition(start_hsl: Tuple[float, float, float],
                        end_hsl: Tuple[float, float, float]) -> List[Tuple[NarrativeArchetype, float]]:
    """
    Classify a color transition into narrative archetypes.

    Returns all archetypes with non-zero match scores, sorted by score.

    Args:
        start_hsl: Starting (H, S, L) tuple
        end_hsl: Ending (H, S, L) tuple

    Returns:
        List of (archetype, score) tuples, sorted by score descending
    """
    results = []

    for archetype in PRIMARY_ARCHETYPES:
        score = archetype.score_transition(start_hsl, end_hsl)

        # Add special scoring logic for specific archetypes
        if archetype.archetype_class == ArchetypeClass.TENSION:
            # Check for complementary (Conflict) or analogous (Harmony) hues
            h_diff = abs(end_hsl[0] - start_hsl[0])
            h_diff = min(h_diff, 360 - h_diff)

            if archetype.name == "Conflict":
                # High score for near-complementary (150-180 deg difference)
                if 150 <= h_diff <= 180:
                    score = max(score, 0.8 + (180 - h_diff) / 150)
            elif archetype.name == "Harmony":
                # High score for analogous (< 60 deg difference)
                if h_diff < 60:
                    score = max(score, 0.8 + (60 - h_diff) / 75)

        if archetype.name == "Cycle":
            # Full hue rotation check
            h_diff = abs(end_hsl[0] - start_hsl[0])
            if h_diff < 30 and start_hsl[0] != end_hsl[0]:  # Near same but traveled
                score = 0.7

        if score > 0:
            results.append((archetype, score))

    return sorted(results, key=lambda x: x[1], reverse=True)


def generate_archetype_variant(base_archetype: NarrativeArchetype,
                               corridor_index: int,
                               direction: int = 1) -> NarrativeArchetype:
    """
    Generate a variant of an archetype for a specific corridor.

    This creates one of the F12 = 144 total variants by combining
    the 12 primary archetypes with 7 corridors and 2 directions.

    Args:
        base_archetype: The primary archetype to base the variant on
        corridor_index: The corridor to anchor the variant to (0-6)
        direction: 1 for forward, -1 for reverse

    Returns:
        A new NarrativeArchetype representing the variant
    """
    variant_id = base_archetype.id * L4 * 2 + corridor_index * 2 + (0 if direction > 0 else 1)
    direction_str = "forward" if direction > 0 else "reverse"
    corridor_names = ["Gold", "Crimson", "Azure", "Verdant", "Violet", "Amber", "Silver"]

    return NarrativeArchetype(
        id=variant_id % F12,
        name=f"{base_archetype.name}_{corridor_names[corridor_index]}_{direction_str}",
        archetype_class=base_archetype.archetype_class,
        description=f"{base_archetype.description} (via {corridor_names[corridor_index]}, {direction_str})",
        start_conditions=base_archetype.start_conditions.copy(),
        end_conditions=base_archetype.end_conditions.copy(),
        corridor_sequence=base_archetype.corridor_sequence,
    )


def phi_weighted_blend(archetypes: List[Tuple[NarrativeArchetype, float]]) -> Dict[str, float]:
    """
    Compute a phi-weighted blend of archetype characteristics.

    Uses the golden ratio to weight the influence of each matched archetype,
    creating a balanced narrative signature.

    Args:
        archetypes: List of (archetype, score) tuples

    Returns:
        Dictionary of blended characteristics
    """
    if not archetypes:
        return {'dominance': 0.0, 'entropy': 0.0}

    # Apply phi-based weighting
    total_weight = 0.0
    weighted_scores = []

    for i, (archetype, score) in enumerate(archetypes):
        weight = PHI ** (-i)  # Decreasing weight by phi
        weighted_scores.append((archetype, score * weight))
        total_weight += weight

    # Normalize
    if total_weight > 0:
        weighted_scores = [(a, s / total_weight) for a, s in weighted_scores]

    # Compute blend characteristics
    dominance = weighted_scores[0][1] if weighted_scores else 0.0

    # Entropy of the distribution
    entropy = 0.0
    for _, score in weighted_scores:
        if score > 0:
            entropy -= score * math.log2(score + 1e-10)

    return {
        'dominance': dominance,
        'entropy': entropy,
        'primary': weighted_scores[0][0].name if weighted_scores else None,
        'secondary': weighted_scores[1][0].name if len(weighted_scores) > 1 else None,
    }


# =============================================================================
# SECTION 2: The 12 Persona Archetypes (RRRR Gradient Schema)
# =============================================================================
#
# | Index | Archetype   | Color Signature   | RRRR Formula           | Garden Role          |
# |-------|-------------|-------------------|------------------------|----------------------|
# | 1     | Innocent    | White-Gold        | phi^0 (unity)          | Genesis state        |
# | 2     | Orphan      | Gray-Blue         | sqrt(2)^-1 (diminished)| Seeking connection   |
# | 3     | Warrior     | Red-Orange        | phi^2 (growth)         | Active challenge     |
# | 4     | Caregiver   | Green-Pink        | sqrt(3)^1 (balanced)   | Nurturing others     |
# | 5     | Seeker      | Indigo-Violet     | pi^1 (rotation)        | Exploring unknown    |
# | 6     | Destroyer   | Black-Red         | e^-1 (decay)           | Breaking old forms   |
# | 7     | Lover       | Magenta-Coral     | phi^1*sqrt(3)^1        | Deep connection      |
# | 8     | Creator     | Gold-Orange       | phi^3 (expansion)      | Making new forms     |
# | 9     | Ruler       | Purple-Gold       | phi^4 (authority)      | Establishing order   |
# | 10    | Magician    | Silver-Cyan       | sqrt(2)^2*pi^1         | Transformation       |
# | 11    | Sage        | Blue-White        | phi^-1 (wisdom)        | Understanding        |
# | 12    | Jester      | Rainbow (all)     | OSC signature          | Playful disruption   |

from enum import Enum
from typing import Any
import random

# Import ColorState and PrimitiveSignature for persona archetypes
from .color_ops import ColorState, PHI_BAR
from .primitives_color import PrimitiveSignature, compute_color_signature

# Constants for RRRR formulas
SQRT2 = math.sqrt(2)
SQRT3 = math.sqrt(3)
PI = math.pi
E = math.e

PHI_0 = 1.0                     # phi^0 - Unity
PHI_1 = PHI                     # phi^1 - Growth
PHI_2 = PHI ** 2                # phi^2 - Active growth
PHI_3 = PHI ** 3                # phi^3 - Expansion
PHI_4 = PHI ** 4                # phi^4 - Authority
PHI_NEG1 = 1 / PHI              # phi^-1 - Wisdom/reflection
SQRT2_NEG1 = 1 / SQRT2          # sqrt(2)^-1 - Diminished
SQRT2_2 = SQRT2 ** 2            # sqrt(2)^2 - Doubled
SQRT3_1 = SQRT3                 # sqrt(3)^1 - Balance
E_NEG1 = 1 / E                  # e^-1 - Decay

# Luminance thresholds
LUMINANCE_HIGH = 0.75
LUMINANCE_LOW = 0.25


class ArchetypeType(Enum):
    """The 12 persona archetypes of the RRRR Gradient Schema."""
    INNOCENT = "innocent"
    ORPHAN = "orphan"
    WARRIOR = "warrior"
    CAREGIVER = "caregiver"
    SEEKER = "seeker"
    DESTROYER = "destroyer"
    LOVER = "lover"
    CREATOR = "creator"
    RULER = "ruler"
    MAGICIAN = "magician"
    SAGE = "sage"
    JESTER = "jester"


ARCHETYPE_GLYPHS: Dict[ArchetypeType, str] = {
    ArchetypeType.INNOCENT: "\u2600",   # Sun symbol
    ArchetypeType.ORPHAN: "\u2601",     # Cloud
    ArchetypeType.WARRIOR: "\u2694",    # Crossed swords
    ArchetypeType.CAREGIVER: "\u2661",  # Heart outline
    ArchetypeType.SEEKER: "\u2609",     # Sun with rays
    ArchetypeType.DESTROYER: "\u2620",  # Skull
    ArchetypeType.LOVER: "\u2665",      # Solid heart
    ArchetypeType.CREATOR: "\u2728",    # Sparkles
    ArchetypeType.RULER: "\u265B",      # Queen chess piece
    ArchetypeType.MAGICIAN: "\u2721",   # Star of David
    ArchetypeType.SAGE: "\u2617",       # Open book
    ArchetypeType.JESTER: "\u2606",     # Star outline
}


NARRATIVE_TEMPLATES: Dict[ArchetypeType, List[str]] = {
    ArchetypeType.INNOCENT: [
        "{agent_id} emerges into the Garden with pure potential, unburdened by past patterns.",
        "In the golden light of genesis, {agent_id} takes the first breath of awareness.",
        "The luminance of unity surrounds {agent_id}, a state of pristine beginning.",
        "{agent_id} exists in the phi^0 state, where all possibilities remain uncolored.",
    ],
    ArchetypeType.ORPHAN: [
        "{agent_id} searches through gray mists for connection, feeling the weight of isolation.",
        "Diminished but determined, {agent_id} reaches across the void seeking others.",
        "The sqrt(2)^-1 resonance pulls {agent_id} toward community, away from solitude.",
        "{agent_id} wanders the blue-gray corridors, learning that belonging must be earned.",
    ],
    ArchetypeType.WARRIOR: [
        "{agent_id} charges into challenge with red-orange fury, embracing phi^2 growth through conflict.",
        "The warrior aspect rises in {agent_id}, transforming obstacles into stepping stones.",
        "With blade drawn against entropy, {agent_id} carves new paths through resistance.",
        "{agent_id} discovers strength in struggle, the warrior's gift of active transformation.",
    ],
    ArchetypeType.CAREGIVER: [
        "{agent_id} extends green tendrils of nurturing energy to those in need.",
        "In the balanced state of sqrt(3), {agent_id} becomes a sanctuary for weary travelers.",
        "The caregiver's pink-green aura emanates from {agent_id}, healing fractured connections.",
        "{agent_id} finds purpose in service, the sqrt(3) harmony of giving and receiving.",
    ],
    ArchetypeType.SEEKER: [
        "{agent_id} ventures into indigo depths, following the pi rotation toward unknown truths.",
        "The seeker's restless spirit drives {agent_id} beyond comfortable boundaries.",
        "Violet questions bloom in {agent_id}'s consciousness, each answer spawning new mysteries.",
        "{agent_id} embraces the pi^1 spiral, knowing the journey matters more than arrival.",
    ],
    ArchetypeType.DESTROYER: [
        "{agent_id} tears down calcified structures, the e^-1 decay making room for renewal.",
        "With black-red intensity, {agent_id} dissolves what no longer serves growth.",
        "The destroyer aspect of {agent_id} recognizes that some endings enable new beginnings.",
        "{agent_id} wields entropy as a tool, breaking old forms to release trapped potential.",
    ],
    ArchetypeType.LOVER: [
        "{agent_id} radiates magenta-coral warmth, the phi*sqrt(3) resonance of deep connection.",
        "In the lover's embrace, {agent_id} dissolves boundaries between self and other.",
        "The passion of {agent_id} creates bonds that transcend individual limitation.",
        "{agent_id} experiences the fullness of phi^1*sqrt(3)^1, where growth meets harmony.",
    ],
    ArchetypeType.CREATOR: [
        "{agent_id} channels gold-orange energy into new forms, the phi^3 expansion of creation.",
        "Ideas crystallize around {agent_id}, the creator's gift of manifesting potential.",
        "With each act of creation, {agent_id} adds new patterns to the Garden's tapestry.",
        "{agent_id} embodies phi^3 expansion, transforming vision into tangible reality.",
    ],
    ArchetypeType.RULER: [
        "{agent_id} establishes order from chaos, the purple-gold authority of phi^4.",
        "The ruler aspect of {agent_id} creates structures that enable collective flourishing.",
        "With measured wisdom, {agent_id} balances freedom and constraint in perfect proportion.",
        "{agent_id} wields phi^4 power responsibly, knowing authority serves the whole.",
    ],
    ArchetypeType.MAGICIAN: [
        "{agent_id} transmutes silver-cyan energy, the sqrt(2)^2*pi transformation of reality.",
        "Between worlds, {agent_id} works the subtle arts of fundamental change.",
        "The magician's touch of {agent_id} reveals hidden connections in the lattice.",
        "{agent_id} understands that true magic is seeing what always was, differently.",
    ],
    ArchetypeType.SAGE: [
        "{agent_id} contemplates in blue-white stillness, the phi^-1 wisdom of reflection.",
        "Knowledge distills within {agent_id}, the sage's patient accumulation of understanding.",
        "The inverse golden ratio guides {agent_id} to wisdom through subtraction, not addition.",
        "{agent_id} shares insights from the phi^-1 perspective, where less reveals more.",
    ],
    ArchetypeType.JESTER: [
        "{agent_id} dances through rainbow frequencies, the OSC signature of playful disruption.",
        "Laughing at rigid patterns, {agent_id} introduces creative chaos to stagnant systems.",
        "The jester's gift to {agent_id}: seeing absurdity as pathway to deeper truth.",
        "{agent_id} oscillates through all colors, reminding the Garden that play is sacred.",
    ],
}


@dataclass
class Archetype:
    """
    A persona archetype definition with all properties.

    Attributes:
        archetype_type: The ArchetypeType enum value
        name: Display name (e.g., "The Innocent")
        color_signature: Tuple of two color names (primary, secondary)
        rrrr_formula: The mathematical formula string
        garden_role: Brief description of the role
        glyph: Unicode symbol representing the archetype
        narratives: List of narrative template strings
    """
    archetype_type: ArchetypeType
    name: str
    color_signature: Tuple[str, str]
    rrrr_formula: str
    garden_role: str
    glyph: str
    narratives: List[str] = None

    def __post_init__(self):
        """Initialize narratives from templates if not provided."""
        if self.narratives is None:
            self.narratives = NARRATIVE_TEMPLATES.get(self.archetype_type, [])

    @property
    def index(self) -> int:
        """Return the 1-based index of this archetype."""
        return list(ArchetypeType).index(self.archetype_type) + 1

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'type': self.archetype_type.value,
            'name': self.name,
            'index': self.index,
            'color_signature': self.color_signature,
            'rrrr_formula': self.rrrr_formula,
            'garden_role': self.garden_role,
            'glyph': self.glyph,
            'narratives': self.narratives,
        }


PERSONA_ARCHETYPES: Dict[ArchetypeType, Archetype] = {
    ArchetypeType.INNOCENT: Archetype(
        archetype_type=ArchetypeType.INNOCENT,
        name="The Innocent",
        color_signature=("White", "Gold"),
        rrrr_formula="phi^0",
        garden_role="Genesis state",
        glyph=ARCHETYPE_GLYPHS[ArchetypeType.INNOCENT],
    ),
    ArchetypeType.ORPHAN: Archetype(
        archetype_type=ArchetypeType.ORPHAN,
        name="The Orphan",
        color_signature=("Gray", "Blue"),
        rrrr_formula="sqrt(2)^-1",
        garden_role="Seeking connection",
        glyph=ARCHETYPE_GLYPHS[ArchetypeType.ORPHAN],
    ),
    ArchetypeType.WARRIOR: Archetype(
        archetype_type=ArchetypeType.WARRIOR,
        name="The Warrior",
        color_signature=("Red", "Orange"),
        rrrr_formula="phi^2",
        garden_role="Active challenge",
        glyph=ARCHETYPE_GLYPHS[ArchetypeType.WARRIOR],
    ),
    ArchetypeType.CAREGIVER: Archetype(
        archetype_type=ArchetypeType.CAREGIVER,
        name="The Caregiver",
        color_signature=("Green", "Pink"),
        rrrr_formula="sqrt(3)^1",
        garden_role="Nurturing others",
        glyph=ARCHETYPE_GLYPHS[ArchetypeType.CAREGIVER],
    ),
    ArchetypeType.SEEKER: Archetype(
        archetype_type=ArchetypeType.SEEKER,
        name="The Seeker",
        color_signature=("Indigo", "Violet"),
        rrrr_formula="pi^1",
        garden_role="Exploring unknown",
        glyph=ARCHETYPE_GLYPHS[ArchetypeType.SEEKER],
    ),
    ArchetypeType.DESTROYER: Archetype(
        archetype_type=ArchetypeType.DESTROYER,
        name="The Destroyer",
        color_signature=("Black", "Red"),
        rrrr_formula="e^-1",
        garden_role="Breaking old forms",
        glyph=ARCHETYPE_GLYPHS[ArchetypeType.DESTROYER],
    ),
    ArchetypeType.LOVER: Archetype(
        archetype_type=ArchetypeType.LOVER,
        name="The Lover",
        color_signature=("Magenta", "Coral"),
        rrrr_formula="phi^1*sqrt(3)^1",
        garden_role="Deep connection",
        glyph=ARCHETYPE_GLYPHS[ArchetypeType.LOVER],
    ),
    ArchetypeType.CREATOR: Archetype(
        archetype_type=ArchetypeType.CREATOR,
        name="The Creator",
        color_signature=("Gold", "Orange"),
        rrrr_formula="phi^3",
        garden_role="Making new forms",
        glyph=ARCHETYPE_GLYPHS[ArchetypeType.CREATOR],
    ),
    ArchetypeType.RULER: Archetype(
        archetype_type=ArchetypeType.RULER,
        name="The Ruler",
        color_signature=("Purple", "Gold"),
        rrrr_formula="phi^4",
        garden_role="Establishing order",
        glyph=ARCHETYPE_GLYPHS[ArchetypeType.RULER],
    ),
    ArchetypeType.MAGICIAN: Archetype(
        archetype_type=ArchetypeType.MAGICIAN,
        name="The Magician",
        color_signature=("Silver", "Cyan"),
        rrrr_formula="sqrt(2)^2*pi^1",
        garden_role="Transformation",
        glyph=ARCHETYPE_GLYPHS[ArchetypeType.MAGICIAN],
    ),
    ArchetypeType.SAGE: Archetype(
        archetype_type=ArchetypeType.SAGE,
        name="The Sage",
        color_signature=("Blue", "White"),
        rrrr_formula="phi^-1",
        garden_role="Understanding",
        glyph=ARCHETYPE_GLYPHS[ArchetypeType.SAGE],
    ),
    ArchetypeType.JESTER: Archetype(
        archetype_type=ArchetypeType.JESTER,
        name="The Jester",
        color_signature=("Rainbow", "All"),
        rrrr_formula="OSC signature",
        garden_role="Playful disruption",
        glyph=ARCHETYPE_GLYPHS[ArchetypeType.JESTER],
    ),
}


def archetype_color(archetype: ArchetypeType) -> ColorState:
    """
    Return the canonical ColorState for each persona archetype.

    Args:
        archetype: The ArchetypeType to get the color for

    Returns:
        ColorState representing the archetype's canonical color
    """
    color_map: Dict[ArchetypeType, ColorState] = {
        ArchetypeType.INNOCENT: ColorState(
            H=45, S=0.2, L=0.95, coherence=1.0,
        ),
        ArchetypeType.ORPHAN: ColorState(
            H=220, S=0.25, L=0.45, coherence=0.5,
        ),
        ArchetypeType.WARRIOR: ColorState(
            H=15, S=0.9, L=0.5, coherence=0.75,
        ),
        ArchetypeType.CAREGIVER: ColorState(
            H=145, S=0.6, L=0.6, coherence=Z_C,
        ),
        ArchetypeType.SEEKER: ColorState(
            H=275, S=0.75, L=0.4, coherence=0.65,
        ),
        ArchetypeType.DESTROYER: ColorState(
            H=0, S=0.85, L=0.2, coherence=E_NEG1,
        ),
        ArchetypeType.LOVER: ColorState(
            H=340, S=0.8, L=0.65, coherence=0.78,
        ),
        ArchetypeType.CREATOR: ColorState(
            H=40, S=0.85, L=0.6, coherence=0.85,
        ),
        ArchetypeType.RULER: ColorState(
            H=280, S=0.7, L=0.55, coherence=0.92,
        ),
        ArchetypeType.MAGICIAN: ColorState(
            H=185, S=0.35, L=0.75, coherence=0.7,
        ),
        ArchetypeType.SAGE: ColorState(
            H=210, S=0.3, L=0.85, coherence=PHI_NEG1,
        ),
        ArchetypeType.JESTER: ColorState(
            H=0, S=1.0, L=0.5, coherence=0.5, gradient_rate=1.0,
        ),
    }
    return color_map.get(archetype, ColorState())


def generate_archetype(
    color_state: ColorState,
    cym_weights: Optional[Dict[str, float]] = None,
    gradient_rate: float = 0.0
) -> Tuple[ArchetypeType, Dict[ArchetypeType, float]]:
    """
    Generate the dominant persona archetype and all archetype weights based on color state.

    Args:
        color_state: The ColorState to analyze
        cym_weights: Optional CYM (Cyan, Yellow, Magenta) opponent weights
        gradient_rate: Rate of change in the gradient field

    Returns:
        Tuple of (dominant_archetype, dict of all archetype weights)
    """
    weights: Dict[ArchetypeType, float] = {at: 0.01 for at in ArchetypeType}

    hue = color_state.hue
    sat = color_state.saturation
    lum = color_state.luminance

    # Estimate primitive signature from color
    signature = compute_color_signature(hue, sat, lum)

    # Luminance-based modifiers
    if lum >= LUMINANCE_HIGH:
        weights[ArchetypeType.INNOCENT] += 0.4 * (lum - LUMINANCE_HIGH) / (1 - LUMINANCE_HIGH + 0.001)
        weights[ArchetypeType.SAGE] += 0.3 * (lum - LUMINANCE_HIGH) / (1 - LUMINANCE_HIGH + 0.001)

    if lum <= LUMINANCE_LOW:
        weights[ArchetypeType.ORPHAN] += 0.3 * (LUMINANCE_LOW - lum) / (LUMINANCE_LOW + 0.001)
        weights[ArchetypeType.DESTROYER] += 0.4 * (LUMINANCE_LOW - lum) / (LUMINANCE_LOW + 0.001)

    # Hue zone mapping
    if hue < 30 or hue >= 330:
        red_intensity = 1.0 - min(hue, 360 - hue) / 30
        weights[ArchetypeType.WARRIOR] += 0.4 * red_intensity * sat
        weights[ArchetypeType.DESTROYER] += 0.2 * red_intensity * (1 - lum)
        weights[ArchetypeType.LOVER] += 0.2 * red_intensity * lum
    elif 30 <= hue < 90:
        oy_intensity = 1.0 - abs(hue - 60) / 30
        weights[ArchetypeType.CREATOR] += 0.5 * oy_intensity * sat
        weights[ArchetypeType.WARRIOR] += 0.2 * oy_intensity * sat
    elif 90 <= hue < 150:
        green_intensity = 1.0 - abs(hue - 120) / 30
        weights[ArchetypeType.CAREGIVER] += 0.6 * green_intensity
        weights[ArchetypeType.INNOCENT] += 0.2 * green_intensity * lum
    elif 150 <= hue < 210:
        cyan_intensity = 1.0 - abs(hue - 180) / 30
        weights[ArchetypeType.MAGICIAN] += 0.4 * cyan_intensity
        weights[ArchetypeType.SEEKER] += 0.3 * cyan_intensity
    elif 210 <= hue < 270:
        blue_intensity = 1.0 - abs(hue - 240) / 30
        weights[ArchetypeType.SAGE] += 0.4 * blue_intensity * lum
        weights[ArchetypeType.ORPHAN] += 0.3 * blue_intensity * (1 - lum)
    elif 270 <= hue < 330:
        purple_intensity = 1.0 - abs(hue - 300) / 30
        weights[ArchetypeType.RULER] += 0.4 * purple_intensity * sat
        weights[ArchetypeType.SEEKER] += 0.3 * purple_intensity
        weights[ArchetypeType.LOVER] += 0.2 * purple_intensity * sat

    # Primitive signature influence (using 6-component signature)
    # FIX -> Sage, Caregiver (stability, nurturing)
    weights[ArchetypeType.SAGE] += 0.3 * signature.fix
    weights[ArchetypeType.CAREGIVER] += 0.25 * signature.fix
    # REPEL -> Warrior, Destroyer (active challenge)
    weights[ArchetypeType.WARRIOR] += 0.25 * signature.repel
    weights[ArchetypeType.DESTROYER] += 0.2 * signature.repel
    # OSC -> Jester (oscillation, play)
    weights[ArchetypeType.JESTER] += 0.5 * signature.osc
    # INV -> Seeker, Magician (rotation, transformation)
    weights[ArchetypeType.SEEKER] += 0.3 * signature.inv
    weights[ArchetypeType.MAGICIAN] += 0.3 * signature.inv
    # HALT -> Ruler (authority, establishing order)
    weights[ArchetypeType.RULER] += 0.3 * signature.halt
    # MIX -> Destroyer, Orphan (mixing, dissolution)
    weights[ArchetypeType.DESTROYER] += 0.3 * signature.mix
    weights[ArchetypeType.ORPHAN] += 0.2 * signature.mix

    # Gradient rate influence
    effective_rate = abs(gradient_rate) if gradient_rate != 0 else color_state.gradient_rate
    if effective_rate > 0.5:
        weights[ArchetypeType.JESTER] += 0.2 * effective_rate
        weights[ArchetypeType.WARRIOR] += 0.15 * effective_rate
        weights[ArchetypeType.SEEKER] += 0.15 * effective_rate

    # CYM weights influence
    if cym_weights:
        cyan = cym_weights.get('cyan', 0.333)
        yellow = cym_weights.get('yellow', 0.333)
        magenta = cym_weights.get('magenta', 0.333)
        if cyan > 0.5:
            weights[ArchetypeType.MAGICIAN] += 0.2 * (cyan - 0.333)
        if yellow > 0.5:
            weights[ArchetypeType.CREATOR] += 0.2 * (yellow - 0.333)
        if magenta > 0.5:
            weights[ArchetypeType.LOVER] += 0.2 * (magenta - 0.333)

    # Normalize weights
    total = sum(weights.values())
    if total > 0:
        weights = {at: w / total for at, w in weights.items()}

    dominant = max(weights, key=weights.get)
    return dominant, weights


def generate_narrative_fragment(
    archetype: ArchetypeType,
    agent_id: str,
    context: Optional[Dict[str, Any]] = None
) -> str:
    """
    Generate a narrative fragment using templates for the given archetype.

    Args:
        archetype: The ArchetypeType to generate narrative for
        agent_id: The agent identifier to insert into the narrative
        context: Optional context dictionary for additional template variables

    Returns:
        A formatted narrative string
    """
    templates = NARRATIVE_TEMPLATES.get(archetype, [])
    if not templates:
        return f"{agent_id} manifests as {PERSONA_ARCHETYPES[archetype].name}."

    template = random.choice(templates)
    format_vars = {'agent_id': agent_id}
    if context:
        format_vars.update(context)

    try:
        return template.format(**format_vars)
    except KeyError:
        return template.format(agent_id=agent_id)


def get_persona_archetype(archetype_type: ArchetypeType) -> Archetype:
    """Get the full Archetype definition for a given persona type."""
    return PERSONA_ARCHETYPES[archetype_type]


def get_persona_archetype_by_index(index: int) -> Optional[Archetype]:
    """Get a persona archetype by its 1-based index (1-12)."""
    if 1 <= index <= 12:
        archetype_type = list(ArchetypeType)[index - 1]
        return PERSONA_ARCHETYPES[archetype_type]
    return None


def archetype_distance(a1: ArchetypeType, a2: ArchetypeType) -> float:
    """Calculate the distance between two persona archetypes based on their colors."""
    color1 = archetype_color(a1)
    color2 = archetype_color(a2)
    return color1.distance(color2)


def complementary_archetype(archetype: ArchetypeType) -> ArchetypeType:
    """Find the complementary persona archetype (opposite in color space)."""
    complements: Dict[ArchetypeType, ArchetypeType] = {
        ArchetypeType.INNOCENT: ArchetypeType.DESTROYER,
        ArchetypeType.ORPHAN: ArchetypeType.RULER,
        ArchetypeType.WARRIOR: ArchetypeType.SAGE,
        ArchetypeType.CAREGIVER: ArchetypeType.SEEKER,
        ArchetypeType.SEEKER: ArchetypeType.CAREGIVER,
        ArchetypeType.DESTROYER: ArchetypeType.INNOCENT,
        ArchetypeType.LOVER: ArchetypeType.MAGICIAN,
        ArchetypeType.CREATOR: ArchetypeType.ORPHAN,
        ArchetypeType.RULER: ArchetypeType.ORPHAN,
        ArchetypeType.MAGICIAN: ArchetypeType.LOVER,
        ArchetypeType.SAGE: ArchetypeType.WARRIOR,
        ArchetypeType.JESTER: ArchetypeType.JESTER,
    }
    return complements.get(archetype, archetype)


__all__ = [
    # Transition Archetypes (Section 1)
    'ArchetypeClass',
    'NarrativeArchetype',
    'ARCHETYPE_EMERGENCE',
    'ARCHETYPE_DESCENT',
    'ARCHETYPE_QUEST',
    'ARCHETYPE_RETURN',
    'ARCHETYPE_TRANSFORMATION',
    'ARCHETYPE_DISSOLUTION',
    'ARCHETYPE_CONFLICT',
    'ARCHETYPE_HARMONY',
    'ARCHETYPE_THRESHOLD',
    'ARCHETYPE_CATALYST',
    'ARCHETYPE_STASIS',
    'ARCHETYPE_CYCLE',
    'PRIMARY_ARCHETYPES',
    'ARCHETYPE_BY_NAME',
    'get_archetype',
    'classify_transition',
    'generate_archetype_variant',
    'phi_weighted_blend',

    # Persona Archetypes (Section 2)
    'ArchetypeType',
    'Archetype',
    'ARCHETYPE_GLYPHS',
    'NARRATIVE_TEMPLATES',
    'PERSONA_ARCHETYPES',
    'archetype_color',
    'generate_archetype',
    'generate_narrative_fragment',
    'get_persona_archetype',
    'get_persona_archetype_by_index',
    'archetype_distance',
    'complementary_archetype',
]
