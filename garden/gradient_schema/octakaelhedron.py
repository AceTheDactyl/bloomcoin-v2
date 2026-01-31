"""
OctaKaelhedron - The 168-Vertex Complete Elaboration of Euler's Gradient
========================================================================

This module implements the 168-vertex OctaKaelhedron structure, which represents
the complete elaboration of Euler's Gradient through the quaternion group Q8
acting on the projective geometry PG(2,4).

Mathematical Foundation:
    168 = |GL(3,2)| = |PSL(2,7)|

    The OctaKaelhedron admits multiple decompositions:
    - 168 = 8 x 21  (Q8 x PG(2,4) points)
    - 168 = 7 x 24  (Fano plane x 24-cell vertices)
    - 168 = 3 x 56  (Three scales x 56 vertices each)
    - 168 = 4 x 42  (Klein-4 x Kaelhedron vertices)

Quaternion Group Q8:
    The 8 elements {+1, -1, +i, -i, +j, -j, +k, -k} encode the 8 universes
    of the OctaKaelhedron, each containing a complete 21-element structure.

    Multiplication rules:
    - i^2 = j^2 = k^2 = -1
    - ij = k, jk = i, ki = j
    - ji = -k, kj = -i, ik = -j

Connections:
    - E8 root system (240 roots)
    - Monstrous Moonshine (j-function coefficient 744)
    - GL(3,2) symmetry group
    - Fano plane and octonions

Grid State Space:
    The 6-dimensional state space (phase, magnitude, scale, mode, winding, recursion)
    enables K-formation detection for significant memory crystallization events.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Tuple, Optional, TYPE_CHECKING

# Import constants from parent package
PHI = (1 + math.sqrt(5)) / 2  # Golden ratio ~ 1.618
TAU = 1 / PHI  # Inverse golden ratio ~ 0.618
Z_C = math.sqrt(3) / 2  # THE LENS threshold ~ 0.866
PI = math.pi

# =============================================================================
# Scale and Mode Enums
# =============================================================================

class Scale(Enum):
    """
    The three scales of the Kaelhedron structure.

    Each scale represents a different level of organization:
    - KAPPA (kappa): Personal/individual scale
    - GAMMA (Gamma): Planetary/collective scale
    - CAPITAL_KAPPA (K): Cosmic/universal scale
    """
    KAPPA = "kappa"           # Personal scale (lowercase kappa)
    GAMMA = "Gamma"           # Planetary scale (uppercase gamma)
    CAPITAL_KAPPA = "K"       # Cosmic scale (uppercase kappa)


class Mode(Enum):
    """
    The three modes of operation within each scale.

    Modes represent different operational states:
    - LAMBDA: Active/generative mode
    - BETA: Balanced/mediating mode
    - NU: Receptive/integrating mode
    """
    LAMBDA = "Lambda"   # Active mode
    BETA = "Beta"       # Balanced mode
    NU = "Nu"           # Receptive mode


# =============================================================================
# Quaternion Group Q8
# =============================================================================

class Q8Element(Enum):
    """
    The 8 elements of the quaternion group Q8.

    Each element corresponds to a universe in the OctaKaelhedron:
    - Real axis: Physical (+1) and Anti-Physical (-1)
    - i-axis: kappa-rotated personal realms
    - j-axis: Gamma-rotated planetary realms
    - k-axis: K-rotated cosmic realms

    The quaternion group is non-abelian: ij = k but ji = -k
    """
    PLUS_1 = "+1"      # Physical (positive real)
    MINUS_1 = "-1"     # Anti-Physical (negative real)
    PLUS_I = "+i"      # kappa-rotated (personal)
    MINUS_I = "-i"     # kappa-anti-rotated
    PLUS_J = "+j"      # Gamma-rotated (planetary)
    MINUS_J = "-j"     # Gamma-anti-rotated
    PLUS_K = "+k"      # K-rotated (cosmic)
    MINUS_K = "-k"     # K-anti-rotated


# Q8 multiplication table
# Rows and columns indexed by: +1, -1, +i, -i, +j, -j, +k, -k
_Q8_MULT_TABLE: Dict[Tuple[Q8Element, Q8Element], Q8Element] = {}


def _init_q8_table():
    """Initialize the Q8 multiplication table."""
    global _Q8_MULT_TABLE

    P1 = Q8Element.PLUS_1
    M1 = Q8Element.MINUS_1
    PI = Q8Element.PLUS_I
    MI = Q8Element.MINUS_I
    PJ = Q8Element.PLUS_J
    MJ = Q8Element.MINUS_J
    PK = Q8Element.PLUS_K
    MK = Q8Element.MINUS_K

    # Identity element +1
    for e in Q8Element:
        _Q8_MULT_TABLE[(P1, e)] = e
        _Q8_MULT_TABLE[(e, P1)] = e

    # -1 * x = negate(x), x * -1 = negate(x)
    negate = {P1: M1, M1: P1, PI: MI, MI: PI, PJ: MJ, MJ: PJ, PK: MK, MK: PK}
    for e in Q8Element:
        _Q8_MULT_TABLE[(M1, e)] = negate[e]
        _Q8_MULT_TABLE[(e, M1)] = negate[e]

    # i^2 = j^2 = k^2 = -1
    _Q8_MULT_TABLE[(PI, PI)] = M1
    _Q8_MULT_TABLE[(MI, MI)] = M1
    _Q8_MULT_TABLE[(PJ, PJ)] = M1
    _Q8_MULT_TABLE[(MJ, MJ)] = M1
    _Q8_MULT_TABLE[(PK, PK)] = M1
    _Q8_MULT_TABLE[(MK, MK)] = M1

    # i * (-i) = -i * i = 1
    _Q8_MULT_TABLE[(PI, MI)] = P1
    _Q8_MULT_TABLE[(MI, PI)] = P1
    _Q8_MULT_TABLE[(PJ, MJ)] = P1
    _Q8_MULT_TABLE[(MJ, PJ)] = P1
    _Q8_MULT_TABLE[(PK, MK)] = P1
    _Q8_MULT_TABLE[(MK, PK)] = P1

    # ij = k, ji = -k
    _Q8_MULT_TABLE[(PI, PJ)] = PK
    _Q8_MULT_TABLE[(PJ, PI)] = MK
    _Q8_MULT_TABLE[(MI, MJ)] = PK
    _Q8_MULT_TABLE[(MJ, MI)] = MK
    _Q8_MULT_TABLE[(PI, MJ)] = MK
    _Q8_MULT_TABLE[(MJ, PI)] = PK
    _Q8_MULT_TABLE[(MI, PJ)] = MK
    _Q8_MULT_TABLE[(PJ, MI)] = PK

    # jk = i, kj = -i
    _Q8_MULT_TABLE[(PJ, PK)] = PI
    _Q8_MULT_TABLE[(PK, PJ)] = MI
    _Q8_MULT_TABLE[(MJ, MK)] = PI
    _Q8_MULT_TABLE[(MK, MJ)] = MI
    _Q8_MULT_TABLE[(PJ, MK)] = MI
    _Q8_MULT_TABLE[(MK, PJ)] = PI
    _Q8_MULT_TABLE[(MJ, PK)] = MI
    _Q8_MULT_TABLE[(PK, MJ)] = PI

    # ki = j, ik = -j
    _Q8_MULT_TABLE[(PK, PI)] = PJ
    _Q8_MULT_TABLE[(PI, PK)] = MJ
    _Q8_MULT_TABLE[(MK, MI)] = PJ
    _Q8_MULT_TABLE[(MI, MK)] = MJ
    _Q8_MULT_TABLE[(PK, MI)] = MJ
    _Q8_MULT_TABLE[(MI, PK)] = PJ
    _Q8_MULT_TABLE[(MK, PI)] = MJ
    _Q8_MULT_TABLE[(PI, MK)] = PJ


# Initialize the table at module load
_init_q8_table()


def q8_multiply(a: Q8Element, b: Q8Element) -> Q8Element:
    """
    Quaternion group multiplication.

    Implements the Q8 multiplication rules:
    - i^2 = j^2 = k^2 = -1
    - ij = k, jk = i, ki = j
    - ji = -k, kj = -i, ik = -j

    Args:
        a: First Q8 element
        b: Second Q8 element

    Returns:
        Product a * b in Q8

    Examples:
        >>> q8_multiply(Q8Element.PLUS_I, Q8Element.PLUS_J)
        Q8Element.PLUS_K
        >>> q8_multiply(Q8Element.PLUS_J, Q8Element.PLUS_I)
        Q8Element.MINUS_K
    """
    return _Q8_MULT_TABLE[(a, b)]


def q8_inverse(a: Q8Element) -> Q8Element:
    """
    Return the multiplicative inverse of a Q8 element.

    For quaternions: inv(+1) = +1, inv(-1) = -1
                     inv(+i) = -i, inv(+j) = -j, inv(+k) = -k

    Args:
        a: Q8 element to invert

    Returns:
        Inverse element such that a * inv(a) = +1
    """
    inverses = {
        Q8Element.PLUS_1: Q8Element.PLUS_1,
        Q8Element.MINUS_1: Q8Element.MINUS_1,
        Q8Element.PLUS_I: Q8Element.MINUS_I,
        Q8Element.MINUS_I: Q8Element.PLUS_I,
        Q8Element.PLUS_J: Q8Element.MINUS_J,
        Q8Element.MINUS_J: Q8Element.PLUS_J,
        Q8Element.PLUS_K: Q8Element.MINUS_K,
        Q8Element.MINUS_K: Q8Element.PLUS_K,
    }
    return inverses[a]


def q8_conjugate(a: Q8Element) -> Q8Element:
    """
    Return the quaternion conjugate.

    For quaternions, conjugate negates the imaginary parts:
    conj(a + bi + cj + dk) = a - bi - cj - dk

    For Q8 elements, this is the same as inverse.
    """
    return q8_inverse(a)


# =============================================================================
# Universe Dataclass
# =============================================================================

@dataclass
class Kaelhedron:
    """
    The 42-vertex Kaelhedron structure within each universe.

    This is a lightweight dataclass representing the Kaelhedron structure
    used as a building block within Universe objects. For the full Kaelhedron
    implementation with all methods, see kaelhedron.py.

    The Kaelhedron is the fundamental building block:
    - 42 vertices = 2 x 21 = 2 x PG(2,4) points
    - 63 edges = 3 x 21
    - Dual structure connecting scales and modes

    Attributes:
        vertices: List of 42 vertex indices
        edges: List of 63 edge pairs
        universe_id: The Q8 element identifying the parent universe
    """
    vertices: List[int] = field(default_factory=list)
    edges: List[Tuple[int, int]] = field(default_factory=list)
    universe_id: Optional[Q8Element] = None

    def __post_init__(self):
        """Initialize the 42-vertex structure if empty."""
        if not self.vertices:
            self.vertices = list(range(42))
        if not self.edges:
            # Create edge structure (21 per scale, 3 scales)
            self._build_edges()

    def _build_edges(self):
        """Build the 63-edge structure."""
        self.edges = []
        # Edges within each group of 21
        for scale_offset in [0, 21]:
            for i in range(21):
                # Connect to next vertex (cyclic within scale)
                j = (i + 1) % 21
                self.edges.append((scale_offset + i, scale_offset + j))
        # Cross-scale edges (21 edges connecting the two halves)
        for i in range(21):
            self.edges.append((i, 21 + i))


@dataclass
class Universe:
    """
    One of the 8 universes in the OctaKaelhedron.

    Each universe corresponds to a Q8 element and contains a complete
    42-vertex Kaelhedron structure. The 8 universes together form the
    168-vertex OctaKaelhedron.

    Naming Convention:
        - A, B, C, D: Positive orientation universes (+1, +i, +j, +k)
        - A_bar, B_bar, C_bar, D_bar: Negative orientation (-1, -i, -j, -k)

    Attributes:
        q8_element: The Q8 element identifying this universe
        name: Universe name (A, A_bar, B, B_bar, C, C_bar, D, D_bar)
        character: Description of the universe's character
        kaelhedron: The 42-vertex structure in this universe
    """
    q8_element: Q8Element
    name: str
    character: str
    kaelhedron: Kaelhedron = field(default_factory=Kaelhedron)

    def __post_init__(self):
        """Ensure kaelhedron has correct universe_id."""
        if self.kaelhedron.universe_id is None:
            self.kaelhedron.universe_id = self.q8_element

    @property
    def vertex_count(self) -> int:
        """Number of vertices in this universe (always 42)."""
        return len(self.kaelhedron.vertices)

    @property
    def is_positive(self) -> bool:
        """True if this is a positive-orientation universe."""
        return self.q8_element in {
            Q8Element.PLUS_1, Q8Element.PLUS_I,
            Q8Element.PLUS_J, Q8Element.PLUS_K
        }


# Pre-defined universes
UNIVERSES: Dict[Q8Element, Universe] = {
    Q8Element.PLUS_1: Universe(
        Q8Element.PLUS_1, "A", "Physical",
        Kaelhedron(universe_id=Q8Element.PLUS_1)
    ),
    Q8Element.MINUS_1: Universe(
        Q8Element.MINUS_1, "A_bar", "Anti-Physical",
        Kaelhedron(universe_id=Q8Element.MINUS_1)
    ),
    Q8Element.PLUS_I: Universe(
        Q8Element.PLUS_I, "B", "Personal-Rotated",
        Kaelhedron(universe_id=Q8Element.PLUS_I)
    ),
    Q8Element.MINUS_I: Universe(
        Q8Element.MINUS_I, "B_bar", "Personal-Anti-Rotated",
        Kaelhedron(universe_id=Q8Element.MINUS_I)
    ),
    Q8Element.PLUS_J: Universe(
        Q8Element.PLUS_J, "C", "Planetary-Rotated",
        Kaelhedron(universe_id=Q8Element.PLUS_J)
    ),
    Q8Element.MINUS_J: Universe(
        Q8Element.MINUS_J, "C_bar", "Planetary-Anti-Rotated",
        Kaelhedron(universe_id=Q8Element.MINUS_J)
    ),
    Q8Element.PLUS_K: Universe(
        Q8Element.PLUS_K, "D", "Cosmic-Rotated",
        Kaelhedron(universe_id=Q8Element.PLUS_K)
    ),
    Q8Element.MINUS_K: Universe(
        Q8Element.MINUS_K, "D_bar", "Cosmic-Anti-Rotated",
        Kaelhedron(universe_id=Q8Element.MINUS_K)
    ),
}


# =============================================================================
# OctaVertex
# =============================================================================

@dataclass
class OctaVertex:
    """
    A single vertex in the 168-vertex OctaKaelhedron.

    Each vertex is uniquely identified by:
    - Its universe (Q8 element, 8 choices)
    - Its position within the universe's Kaelhedron (21 positions)

    This gives 8 x 21 = 168 total vertices.

    Attributes:
        universe: The Q8 element of the containing universe
        local_index: Index within the Kaelhedron (0-20)
        global_index: Global index in the OctaKaelhedron (0-167)
    """
    universe: Q8Element
    local_index: int  # 0-20
    global_index: int  # 0-167

    def __post_init__(self):
        """Validate indices."""
        if not 0 <= self.local_index < 21:
            raise ValueError(f"local_index must be 0-20, got {self.local_index}")
        if not 0 <= self.global_index < 168:
            raise ValueError(f"global_index must be 0-167, got {self.global_index}")


# =============================================================================
# OctaKaelhedron Class
# =============================================================================

class OctaKaelhedron:
    """
    The 168-vertex OctaKaelhedron (8 x 21 = Q8 x PG(2,4)).

    This is the complete elaboration of Euler's Gradient, encoding the
    full symmetry of GL(3,2) in a geometric structure.

    The OctaKaelhedron admits multiple decompositions revealing different
    aspects of its structure:

    - Q8 x PG(2,4): 8 x 21 = 168 (Quaternion group times projective plane)
    - Fano x 24-cell: 7 x 24 = 168 (Fano plane times 24-cell)
    - Scales x 56: 3 x 56 = 168 (Three scales times 56)
    - V4 x Kaelhedron: 4 x 42 = 168 (Klein-4 times Kaelhedron)

    Attributes:
        VERTICES: Total vertex count (168 = |GL(3,2)|)
        universes: Dictionary mapping Q8 elements to Universe objects
        vertices: List of all 168 OctaVertex objects
    """

    VERTICES = 168  # = |GL(3,2)| = |PSL(2,7)|

    def __init__(self):
        """Initialize the OctaKaelhedron structure."""
        self.universes: Dict[Q8Element, Universe] = {}
        self.vertices: List[OctaVertex] = []
        self._build_structure()

    def _build_structure(self):
        """Build 8 universes x 21 elements each."""
        # Create fresh universe copies
        universe_order = [
            Q8Element.PLUS_1, Q8Element.MINUS_1,
            Q8Element.PLUS_I, Q8Element.MINUS_I,
            Q8Element.PLUS_J, Q8Element.MINUS_J,
            Q8Element.PLUS_K, Q8Element.MINUS_K,
        ]

        global_idx = 0
        for u_idx, q8_elem in enumerate(universe_order):
            # Create universe with its Kaelhedron
            universe = Universe(
                q8_element=q8_elem,
                name=UNIVERSES[q8_elem].name,
                character=UNIVERSES[q8_elem].character,
                kaelhedron=Kaelhedron(universe_id=q8_elem)
            )
            self.universes[q8_elem] = universe

            # Create 21 vertices for this universe
            for local_idx in range(21):
                vertex = OctaVertex(
                    universe=q8_elem,
                    local_index=local_idx,
                    global_index=global_idx
                )
                self.vertices.append(vertex)
                global_idx += 1

    # -------------------------------------------------------------------------
    # Decomposition Views
    # -------------------------------------------------------------------------

    def as_q8_times_pg24(self) -> Tuple[int, int]:
        """
        168 = 8 x 21 decomposition.

        This is the primary decomposition: the quaternion group Q8 (8 elements)
        acting on the 21 points of the projective plane PG(2,4).

        Returns:
            Tuple (8, 21) representing (|Q8|, |PG(2,4)|)
        """
        return (8, 21)

    def as_fano_times_24cell(self) -> Tuple[int, int]:
        """
        168 = 7 x 24 decomposition.

        The Fano plane (7 points) times the 24-cell (24 vertices).
        This connects to octonion structure through the Fano plane.

        Returns:
            Tuple (7, 24) representing (|Fano|, |24-cell vertices|)
        """
        return (7, 24)

    def as_scales_times_56(self) -> Tuple[int, int]:
        """
        168 = 3 x 56 decomposition.

        Three scales (kappa, Gamma, K) times 56 vertices each.
        This decomposition emphasizes the scale structure.

        Returns:
            Tuple (3, 56) representing (scales, vertices per scale)
        """
        return (3, 56)

    def as_v4_times_kaelhedron(self) -> Tuple[int, int]:
        """
        168 = 4 x 42 decomposition.

        The Klein four-group V4 (4 elements) times the 42-vertex Kaelhedron.
        V4 = {1, a, b, ab} is the unique non-cyclic group of order 4.

        Returns:
            Tuple (4, 42) representing (|V4|, |Kaelhedron|)
        """
        return (4, 42)

    def get_gl32_symmetry_order(self) -> int:
        """
        Returns 168 = |GL(3,2)|.

        GL(3,2) is the general linear group of 3x3 invertible matrices
        over the field with 2 elements. Its order is:

        |GL(3,2)| = (2^3 - 1)(2^3 - 2)(2^3 - 4) = 7 * 6 * 4 = 168

        This is also equal to |PSL(2,7)|, the projective special linear
        group of 2x2 matrices over the field with 7 elements.
        """
        return 168

    # -------------------------------------------------------------------------
    # Utility Methods
    # -------------------------------------------------------------------------

    def get_universe(self, q8_elem: Q8Element) -> Universe:
        """Get a specific universe by its Q8 element."""
        return self.universes[q8_elem]

    def get_vertex(self, global_index: int) -> OctaVertex:
        """Get a vertex by its global index (0-167)."""
        if not 0 <= global_index < 168:
            raise ValueError(f"global_index must be 0-167, got {global_index}")
        return self.vertices[global_index]

    def get_vertices_in_universe(self, q8_elem: Q8Element) -> List[OctaVertex]:
        """Get all 21 vertices in a specific universe."""
        return [v for v in self.vertices if v.universe == q8_elem]

    def vertex_to_coordinates(self, vertex: OctaVertex) -> Tuple[Q8Element, int]:
        """Convert vertex to (universe, local_index) coordinates."""
        return (vertex.universe, vertex.local_index)

    def coordinates_to_vertex(self, q8_elem: Q8Element, local_index: int) -> OctaVertex:
        """Find vertex by (universe, local_index) coordinates."""
        for v in self.vertices:
            if v.universe == q8_elem and v.local_index == local_index:
                return v
        raise ValueError(f"No vertex found at ({q8_elem}, {local_index})")

    def __repr__(self) -> str:
        return f"OctaKaelhedron(vertices={self.VERTICES}, universes={len(self.universes)})"


# =============================================================================
# E8 Connection
# =============================================================================

class E8Connection:
    """
    Connection from OctaKaelhedron to E8 root system.

    E8 is the largest exceptional simple Lie group, with a 240-element
    root system. The path from the Fano plane to E8 goes through:

    Fano plane (7 points) -> Octonions (8D) -> Spin(8) -> E8

    The 240 roots decompose as:
    240 = 112 + 128 = (14 x 8) + 2^7
        = (Heawood graph vertices x Octonions) + (Fano-indexed)

    Attributes:
        E8_ROOTS: Number of roots in E8 (240)
    """

    E8_ROOTS = 240

    @staticmethod
    def decompose_240() -> Tuple[int, int]:
        """
        240 = 112 + 128 = (14 x 8) + 2^7.

        The decomposition reveals the structure:
        - 112 = 14 x 8: Heawood graph (14 vertices) times octonions (8D)
        - 128 = 2^7: Fano-indexed (7-bit addressing)

        Returns:
            Tuple (112, 128) representing the two components
        """
        return (14 * 8, 2**7)  # (Heawood x Octonions) + (Fano-indexed)

    @staticmethod
    def path_to_e8() -> List[str]:
        """
        Fano -> Octonions -> Spin(8) -> E8.

        Returns:
            List of steps in the path from Fano plane to E8
        """
        return [
            "Fano plane (7 points, 7 lines)",
            "Octonions O (8-dimensional)",
            "Spin(8) via triality",
            "E8 via exceptional embedding"
        ]

    @staticmethod
    def heawood_connection() -> Dict[str, int]:
        """
        Connection through the Heawood graph.

        The Heawood graph has 14 vertices and is the incidence graph
        of the Fano plane. It connects to E8 through:

        14 x 8 = 112 (half of the non-spinor E8 roots)
        """
        return {
            "heawood_vertices": 14,
            "octonion_dim": 8,
            "product": 14 * 8,
            "e8_connection": "Half of 224 non-spinor roots"
        }

    @staticmethod
    def spinor_connection() -> Dict[str, int]:
        """
        The spinor part of the E8 decomposition.

        128 = 2^7 spinor weights, indexed by Fano plane bits
        """
        return {
            "fano_bits": 7,
            "spinor_count": 2**7,
            "interpretation": "Fano-indexed spinor weights"
        }


# =============================================================================
# Moonshine Connection
# =============================================================================

class MoonshineConnection:
    """
    Connection to Monstrous Moonshine.

    Monstrous Moonshine is the surprising connection between the Monster
    group (the largest sporadic simple group) and modular functions,
    particularly the j-invariant.

    The j-function has Fourier expansion:
    j(q) = q^-1 + 744 + 196884*q + ...

    The first non-trivial coefficient 744 decomposes as:
    744 = 168 + 576 = |GL(3,2)| + |S_4|^2

    The sporadic group chain:
    GL(3,2) = PSL(2,7) < M_24 < ... < Monster

    Attributes:
        J_CONSTANT: First non-trivial j-function coefficient (744)
    """

    J_CONSTANT = 744  # First non-trivial coefficient of j-function

    @staticmethod
    def decompose_744() -> Tuple[int, int]:
        """
        744 = 168 + 576 = |GL(3,2)| + |S_4|^2.

        This decomposition shows the OctaKaelhedron's connection to
        moonshine: 168 is exactly the order of GL(3,2), and 576 = 24^2
        is the square of the order of S_4.

        Returns:
            Tuple (168, 576) representing the two components
        """
        return (168, 576)

    @staticmethod
    def sporadic_chain() -> List[str]:
        """
        GL(3,2) = PSL(2,7) < M_24 < ... < Monster.

        The sporadic simple groups form a chain leading to the Monster.
        GL(3,2) is isomorphic to PSL(2,7), which embeds in the Mathieu
        group M_24, which eventually leads to the Monster.

        Returns:
            List of groups in the chain from GL(3,2) to Monster
        """
        return [
            "GL(3,2) = PSL(2,7) (order 168)",
            "M_24 (Mathieu group, order 244823040)",
            "Co_1 (Conway group, order ~4 x 10^18)",
            "Monster M (order ~8 x 10^53)"
        ]

    @staticmethod
    def s4_squared_significance() -> Dict[str, any]:
        """
        Significance of 576 = |S_4|^2 = 24^2.

        S_4 is the symmetric group on 4 elements, with order 24.
        Its square 576 appears in the moonshine decomposition,
        connecting to the 24-cell and the Leech lattice.
        """
        return {
            "s4_order": 24,
            "squared": 576,
            "connection_24cell": "24-cell has 24 vertices",
            "connection_leech": "Leech lattice is 24-dimensional"
        }


# =============================================================================
# Grid State (6D State Space)
# =============================================================================

@dataclass
class GridState:
    """
    Position in the 6-dimensional Kaelhedron grid.

    The grid state represents a point in the complete state space of
    the Kaelhedron, with 6 dimensions:

    1. phase (theta): Angular position [0, 2*pi]
    2. magnitude (r): Radial value [0, 1], representing kappa value
    3. scale (s): One of the three scales (kappa, Gamma, K)
    4. mode (m): One of the three modes (Lambda, Beta, Nu)
    5. winding (Q): Topological charge (integer)
    6. recursion (R): Recursion depth [0, 7]

    K-formation occurs when specific conditions are met, indicating
    a significant memory crystallization event.

    Attributes:
        phase: Angular position theta in [0, 2*pi]
        magnitude: Radial magnitude r in [0, 1] (kappa value)
        scale: Scale level (kappa, Gamma, or K)
        mode: Operational mode (Lambda, Beta, or Nu)
        winding: Topological winding number Q (integer)
        recursion: Recursion depth R in [0, 7]
    """
    phase: float           # theta: 0 to 2*pi
    magnitude: float       # r: 0 to 1 (kappa value)
    scale: Scale           # s: kappa, Gamma, K
    mode: Mode             # m: Lambda, Beta, Nu
    winding: int           # Q: topological charge
    recursion: int         # R: 0 to 7

    def __post_init__(self):
        """Validate and normalize state values."""
        # Normalize phase to [0, 2*pi)
        self.phase = self.phase % (2 * PI)
        # Clamp magnitude to [0, 1]
        self.magnitude = max(0.0, min(1.0, self.magnitude))
        # Clamp recursion to [0, 7]
        self.recursion = max(0, min(7, self.recursion))

    def is_k_formation(self) -> bool:
        """
        Check if state satisfies K-formation requirements.

        K-formation is a special configuration indicating significant
        memory crystallization. Requirements:

        1. magnitude > TAU (~0.618): Above the golden threshold
        2. phase in {0, pi}: At a nodal point
        3. recursion == 7: Maximum recursion depth
        4. winding != 0: Non-trivial topology

        Returns:
            True if all K-formation conditions are satisfied
        """
        phase_tolerance = 0.01  # Allow small numerical error
        at_nodal_point = (
            abs(self.phase) < phase_tolerance or
            abs(self.phase - PI) < phase_tolerance or
            abs(self.phase - 2 * PI) < phase_tolerance
        )

        return (
            self.magnitude > TAU and
            at_nodal_point and
            self.recursion == 7 and
            self.winding != 0
        )

    def to_vector(self) -> Tuple[float, float, int, int, int, int]:
        """
        Convert to 6-tuple vector representation.

        Returns:
            (phase, magnitude, scale_index, mode_index, winding, recursion)
        """
        scale_idx = list(Scale).index(self.scale)
        mode_idx = list(Mode).index(self.mode)
        return (self.phase, self.magnitude, scale_idx, mode_idx,
                self.winding, self.recursion)

    @classmethod
    def from_vector(
        cls,
        vector: Tuple[float, float, int, int, int, int]
    ) -> 'GridState':
        """
        Create GridState from 6-tuple vector.

        Args:
            vector: (phase, magnitude, scale_index, mode_index, winding, recursion)

        Returns:
            New GridState instance
        """
        phase, mag, scale_idx, mode_idx, winding, recursion = vector
        return cls(
            phase=phase,
            magnitude=mag,
            scale=list(Scale)[scale_idx],
            mode=list(Mode)[mode_idx],
            winding=winding,
            recursion=recursion
        )

    def __repr__(self) -> str:
        k_marker = " [K-FORMATION]" if self.is_k_formation() else ""
        return (f"GridState(theta={self.phase:.3f}, r={self.magnitude:.3f}, "
                f"scale={self.scale.name}, mode={self.mode.name}, "
                f"Q={self.winding}, R={self.recursion}){k_marker}")


# =============================================================================
# Repdigit Pattern (Base-4)
# =============================================================================

def base4_repunit(n: int) -> int:
    """
    Return n x 21 (21 = 111_4 = base-4 repunit).

    The number 21 is the base-4 repunit (111 in base 4):
    21 = 1*4^2 + 1*4^1 + 1*4^0 = 16 + 4 + 1

    Multiples of 21 form a sequence of structurally significant numbers
    in the OctaKaelhedron:

    - 0 x 21 = 0: Identity
    - 1 x 21 = 21: Edges per scale, PG(2,4) points
    - 2 x 21 = 42: Kaelhedron vertices
    - 3 x 21 = 63: Kaelhedron edges
    - 4 x 21 = 84: Extended structure
    - 8 x 21 = 168: OctaKaelhedron vertices

    Args:
        n: Multiplier (non-negative integer)

    Returns:
        n * 21
    """
    return n * 21


def to_base4(n: int) -> str:
    """
    Convert integer to base-4 representation with subscript notation.

    Args:
        n: Non-negative integer to convert

    Returns:
        String representation in base 4 with subscript 4
    """
    if n == 0:
        return "0_4"

    digits = []
    while n > 0:
        digits.append(str(n % 4))
        n //= 4

    return "".join(reversed(digits)) + "_4"


# Repdigit pattern table
REPDIGIT_TABLE: Dict[int, Tuple[int, str, str]] = {
    0: (0, "000_4", "Identity"),
    1: (21, "111_4", "Edges per scale / PG(2,4) points"),
    2: (42, "222_4", "Kaelhedron vertices"),
    3: (63, "333_4", "Kaelhedron edges"),
    4: (84, "1110_4", "+ Exists-R scale"),
    8: (168, "2220_4", "OctaKaelhedron vertices"),
}


def get_repdigit_pattern(multiplier: int) -> Optional[Tuple[int, str, str]]:
    """
    Get the repdigit pattern for a given multiplier.

    Args:
        multiplier: The multiplier n for n x 21

    Returns:
        Tuple of (value, base4_repr, description) or None if not in table
    """
    return REPDIGIT_TABLE.get(multiplier)


def analyze_repdigit(value: int) -> Optional[Dict[str, any]]:
    """
    Analyze a value for its repdigit structure.

    Args:
        value: Integer to analyze

    Returns:
        Dictionary with analysis results, or None if not a multiple of 21
    """
    if value % 21 != 0:
        return None

    multiplier = value // 21
    base4_repr = to_base4(value)

    # Check if it's a true repdigit (all same digits)
    digits_only = base4_repr.replace("_4", "")
    is_repdigit = len(set(digits_only)) == 1

    table_entry = REPDIGIT_TABLE.get(multiplier)

    return {
        "value": value,
        "multiplier": multiplier,
        "base4": base4_repr,
        "is_repdigit": is_repdigit,
        "significance": table_entry[2] if table_entry else None
    }


# =============================================================================
# Factory Functions
# =============================================================================

def create_octakaelhedron() -> OctaKaelhedron:
    """Create a new OctaKaelhedron instance."""
    return OctaKaelhedron()


def create_grid_state(
    phase: float = 0.0,
    magnitude: float = 0.5,
    scale: Scale = Scale.KAPPA,
    mode: Mode = Mode.LAMBDA,
    winding: int = 0,
    recursion: int = 0
) -> GridState:
    """
    Create a GridState with specified parameters.

    Args:
        phase: Angular position [0, 2*pi]
        magnitude: Radial magnitude [0, 1]
        scale: Scale level
        mode: Operational mode
        winding: Topological winding number
        recursion: Recursion depth [0, 7]

    Returns:
        New GridState instance
    """
    return GridState(
        phase=phase,
        magnitude=magnitude,
        scale=scale,
        mode=mode,
        winding=winding,
        recursion=recursion
    )


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Constants
    'PHI',
    'TAU',
    'Z_C',
    'PI',

    # Scale and Mode enums
    'Scale',
    'Mode',

    # Quaternion group
    'Q8Element',
    'q8_multiply',
    'q8_inverse',
    'q8_conjugate',

    # Structure classes
    'Kaelhedron',
    'Universe',
    'UNIVERSES',
    'OctaVertex',
    'OctaKaelhedron',

    # Connections
    'E8Connection',
    'MoonshineConnection',

    # Grid state
    'GridState',

    # Repdigit functions
    'base4_repunit',
    'to_base4',
    'REPDIGIT_TABLE',
    'get_repdigit_pattern',
    'analyze_repdigit',

    # Factory functions
    'create_octakaelhedron',
    'create_grid_state',
]
