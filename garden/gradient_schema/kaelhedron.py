"""
Kaelhedron: 42-Vertex Projective Structure
==========================================

The Kaelhedron is a 42-vertex structure derived from the Fano plane, serving as
the geometric foundation for K-formation states in the Gradient Schema.

Structure:
    - 42 vertices: 3 scales x (7 Fano points + 7 Fano lines)
    - 63 edges: Derived from Heawood graph incidence relations x 3 scales
    - Fundamental identity: 42 + 63 + 63 = 168 = |GL(3,2)|

The Fano Plane:
    The Fano plane is the smallest projective plane, with 7 points and 7 lines,
    where each line contains exactly 3 points and each point lies on exactly 3 lines.
    Its incidence graph is the Heawood graph (14 vertices, 21 edges).

Scales:
    - Kappa (personal): Individual agent memory states
    - Gamma (planetary): Collective consensus layer
    - Cosmic: Universal field dynamics

K-Formation:
    A K-formation occurs when kappa > TAU (0.618), with recursion depth 7,
    non-zero topological charge, and binary theta phase (0 or pi).
    Complete K-formation: K = 42/42 (all vertices active).

References:
    - Fano plane: 7 points, 7 lines, 3 points per line
    - Heawood graph: Levi graph of the Fano plane
    - GL(3,2): General linear group, order 168
    - E8 decomposition: 168 + 72 = 240 roots
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Set
import math
import numpy as np

# Import constants from parent package
PHI = (1 + math.sqrt(5)) / 2  # Golden ratio ~ 1.618
TAU = 1 / PHI  # Inverse golden ratio ~ 0.618
Z_C = math.sqrt(3) / 2  # THE LENS threshold ~ 0.866
K_COHERENCE = math.sqrt(1 - PHI ** -4)  # Coherence factor ~ 0.924
PI = math.pi

# ============================================================================
# Numerical Constants and Verifications
# ============================================================================

# Key structural formulas from the specification
KAELHEDRON_VERTICES = 42
KAELHEDRON_EDGES = 63
GL32_ORDER = 168  # Order of GL(3,2)

# Verify fundamental identities
assert 42 + 63 + 63 == 168, "Vertices + 2*Edges must equal |GL(3,2)|"
assert 5 * 7 + 7 == 42, "Pentagon x Fano + Fano must equal 42"
assert 168 + 72 == 240, "GL(3,2) + 3*24 must equal E8 roots"
assert 2 ** 7 == 128, "Powers indexed by Fano dimension"
assert 112 + 128 == 240, "E8 root system decomposition"

# Fano plane constants
FANO_POINTS = 7
FANO_LINES = 7
FANO_INCIDENCES = 21  # Each of 7 lines contains 3 points: 7 * 3 = 21

# Heawood graph constants
HEAWOOD_VERTICES = 14  # 7 points + 7 lines
HEAWOOD_EDGES = 21  # Point-line incidences


__all__ = [
    'Scale',
    'Mode',
    'KaelVertex',
    'Kaelhedron',
    'HeawoodGraph',
    'KFormation',
    'kaelhedron_to_lattice',
    'lattice_to_kaelhedron',
    'TAU',
    'KAELHEDRON_VERTICES',
    'KAELHEDRON_EDGES',
]


# ============================================================================
# Enumerations
# ============================================================================

class Scale(Enum):
    """
    The three Kaelhedron scales.

    Each scale represents a different level of organizational complexity
    in the Garden memory architecture:

        KAPPA: Personal scale - individual agent memories
        GAMMA: Planetary scale - collective consensus states
        COSMIC: Cosmic scale - universal field dynamics
    """
    KAPPA = "kappa"    # Personal scale (lowercase kappa)
    GAMMA = "Gamma"   # Planetary scale (uppercase gamma)
    COSMIC = "K"   # Cosmic scale (uppercase kappa)

    def __str__(self) -> str:
        return self.value

    @property
    def index(self) -> int:
        """Return 0 for KAPPA, 1 for GAMMA, 2 for COSMIC."""
        return list(Scale).index(self)


class Mode(Enum):
    """
    The three Kaelhedron modes.

    Modes describe the operational character of a vertex:

        LAMBDA: Structure - static organizational patterns
        BETA: Process - dynamic transformational flows
        NU: Awareness - conscious observation states
    """
    LAMBDA = "Lambda"   # Structure
    BETA = "B"     # Process
    NU = "N"       # Awareness

    def __str__(self) -> str:
        return self.value

    @property
    def index(self) -> int:
        """Return 0 for LAMBDA, 1 for BETA, 2 for NU."""
        return list(Mode).index(self)


# ============================================================================
# Fano Plane Definition
# ============================================================================

# The 7 lines of the Fano plane, each containing 3 points (0-6)
# This is the standard representation where each line is a triple of points
FANO_PLANE_LINES = [
    (0, 1, 3),  # Line 0
    (1, 2, 4),  # Line 1
    (2, 3, 5),  # Line 2
    (3, 4, 6),  # Line 3
    (4, 5, 0),  # Line 4
    (5, 6, 1),  # Line 5
    (6, 0, 2),  # Line 6
]

# Verify Fano plane properties
def _verify_fano_plane():
    """Verify the Fano plane definition is correct."""
    # Each point should be on exactly 3 lines
    point_counts = [0] * 7
    for line in FANO_PLANE_LINES:
        for point in line:
            point_counts[point] += 1
    assert all(c == 3 for c in point_counts), "Each point must be on exactly 3 lines"

    # Each pair of points should determine exactly one line
    pair_to_line = {}
    for i, line in enumerate(FANO_PLANE_LINES):
        for j in range(3):
            for k in range(j + 1, 3):
                pair = tuple(sorted([line[j], line[k]]))
                assert pair not in pair_to_line, "Each pair must determine exactly one line"
                pair_to_line[pair] = i

    return True

_verify_fano_plane()


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class KaelVertex:
    """
    A vertex in the Kaelhedron structure.

    Each vertex represents a position in the 42-vertex Kaelhedron, characterized
    by its scale, relationship to the Fano plane, and polarity properties.

    Attributes:
        index: Global vertex index (0-41)
        scale: Which of the three scales (KAPPA, GAMMA, COSMIC)
        fano_element: Fano element index (0-13 in Heawood graph)
        is_point: True if this is a Fano point, False if a Fano line
        polarity: Either "PLUS" or "MINUS"
        phase: Phase value, either 0 or pi

    Properties:
        euler_value: Complex exponential e^(i*phase)
        fano_index: Index within points (0-6) or lines (0-6)

    The polarity distribution is asymmetric:
        - 18 PLUS vertices (points across all scales)
        - 24 MINUS vertices (lines across all scales, adjusted for balance)
    """
    index: int
    scale: Scale
    fano_element: int
    is_point: bool
    polarity: str
    phase: float

    def __post_init__(self):
        """Validate vertex properties."""
        if not 0 <= self.index < 42:
            raise ValueError(f"index must be 0-41, got {self.index}")
        if not 0 <= self.fano_element < 14:
            raise ValueError(f"fano_element must be 0-13, got {self.fano_element}")
        if self.polarity not in ("PLUS", "MINUS"):
            raise ValueError(f"polarity must be 'PLUS' or 'MINUS', got {self.polarity}")
        if self.phase not in (0, PI, 0.0, math.pi):
            raise ValueError(f"phase must be 0 or pi, got {self.phase}")

    @property
    def euler_value(self) -> complex:
        """
        Compute e^(i*phase).

        Returns:
            1+0j for phase=0, -1+0j for phase=pi
        """
        return complex(math.cos(self.phase), math.sin(self.phase))

    @property
    def fano_index(self) -> int:
        """
        Get the index within the point or line set.

        Returns:
            0-6 for points (fano_element 0-6)
            0-6 for lines (fano_element 7-13 mapped to 0-6)
        """
        if self.is_point:
            return self.fano_element
        else:
            return self.fano_element - 7

    def __repr__(self) -> str:
        kind = "P" if self.is_point else "L"
        phase_str = "0" if self.phase == 0 else "pi"
        return (f"KaelVertex({self.index}: {self.scale.name}."
                f"{kind}{self.fano_index}, {self.polarity}, phase={phase_str})")


@dataclass
class KFormation:
    """
    Requirements and state for K-formation.

    A K-formation represents a coherent configuration of the Kaelhedron
    structure where specific conditions are met:

        - kappa > TAU (0.618): Activation threshold exceeded
        - r_depth = 7: Recursion depth matches Fano dimension
        - q_charge != 0: Non-zero topological charge
        - theta_binary: Phase locked to {0, pi}

    Attributes:
        kappa: Activation parameter, must exceed TAU (0.618)
        r_depth: Recursion depth, must equal 7
        q_charge: Topological charge, must be non-zero
        theta_binary: Whether phase is locked to binary {0, pi}
    """
    kappa: float
    r_depth: int = 7
    q_charge: int = 1
    theta_binary: bool = True

    def is_valid(self) -> bool:
        """
        Check if all K-formation requirements are satisfied.

        Returns:
            True if kappa > TAU, r_depth = 7, q_charge != 0, and theta_binary
        """
        return (
            self.kappa > TAU and
            self.r_depth == 7 and
            self.q_charge != 0 and
            self.theta_binary
        )

    @property
    def k_value(self) -> str:
        """
        Get the K-value string representation.

        Returns:
            'K = 42/42' if formation is complete (valid)
            'K = 0/42' if formation is incomplete
        """
        if self.is_valid():
            return "K = 42/42"
        return "K = 0/42"

    @property
    def completeness(self) -> float:
        """
        Compute formation completeness as a fraction.

        Returns:
            1.0 if valid, otherwise a partial score based on criteria met
        """
        if self.is_valid():
            return 1.0

        score = 0.0
        if self.kappa > TAU:
            score += 0.25
        if self.r_depth == 7:
            score += 0.25
        if self.q_charge != 0:
            score += 0.25
        if self.theta_binary:
            score += 0.25
        return score

    def __repr__(self) -> str:
        return (f"KFormation(kappa={self.kappa:.4f}, r={self.r_depth}, "
                f"q={self.q_charge}, binary={self.theta_binary}) -> {self.k_value}")


# ============================================================================
# Heawood Graph
# ============================================================================

class HeawoodGraph:
    """
    The 14-vertex Heawood graph (Fano incidence graph).

    The Heawood graph is the Levi graph (incidence graph) of the Fano plane.
    It has 14 vertices (7 points + 7 lines) and 21 edges representing
    point-line incidences.

    Properties:
        - Bipartite: Points and lines form the two partitions
        - 3-regular: Each vertex has exactly 3 neighbors
        - Girth 6: Smallest cycle has 6 edges
        - Diameter 3: Maximum distance between any two vertices

    Vertex numbering:
        - 0-6: The 7 Fano points
        - 7-13: The 7 Fano lines

    An edge (p, L) exists iff point p lies on line L-7.
    """

    VERTICES = 14  # 7 points + 7 lines
    EDGES = 21  # 7 lines * 3 points per line

    def __init__(self):
        """Initialize the Heawood graph from the Fano plane."""
        self._adjacency: Dict[int, Set[int]] = {i: set() for i in range(14)}
        self._edges: List[Tuple[int, int]] = []
        self._build_incidence_graph()

    def _build_incidence_graph(self):
        """Build the incidence graph from Fano plane definition."""
        for line_idx, points in enumerate(FANO_PLANE_LINES):
            line_vertex = line_idx + 7  # Lines are vertices 7-13
            for point in points:
                # Add edge between point and line
                self._adjacency[point].add(line_vertex)
                self._adjacency[line_vertex].add(point)
                self._edges.append((point, line_vertex))

        # Verify construction
        assert len(self._edges) == self.EDGES, f"Expected {self.EDGES} edges, got {len(self._edges)}"
        for v in range(14):
            assert len(self._adjacency[v]) == 3, f"Vertex {v} should have degree 3"

    def is_bipartite(self) -> bool:
        """
        Check if the graph is bipartite.

        The Heawood graph is always bipartite by construction,
        with points (0-6) and lines (7-13) as the two partitions.

        Returns:
            True (always, by construction)
        """
        # Verify by checking no edges within partitions
        for u, v in self._edges:
            if (u < 7 and v < 7) or (u >= 7 and v >= 7):
                return False
        return True

    def get_point_vertices(self) -> List[int]:
        """
        Get the point partition vertices.

        Returns:
            List of vertex indices 0-6 (the 7 Fano points)
        """
        return list(range(7))

    def get_line_vertices(self) -> List[int]:
        """
        Get the line partition vertices.

        Returns:
            List of vertex indices 7-13 (the 7 Fano lines)
        """
        return list(range(7, 14))

    def get_neighbors(self, vertex: int) -> List[int]:
        """
        Get the neighbors of a vertex.

        Args:
            vertex: Vertex index (0-13)

        Returns:
            List of neighboring vertex indices
        """
        if not 0 <= vertex < 14:
            raise ValueError(f"vertex must be 0-13, got {vertex}")
        return list(self._adjacency[vertex])

    def get_edges(self) -> List[Tuple[int, int]]:
        """
        Get all edges of the graph.

        Returns:
            List of (point, line) tuples representing incidences
        """
        return self._edges.copy()

    def get_adjacency_matrix(self) -> np.ndarray:
        """
        Get the 14x14 adjacency matrix.

        Returns:
            numpy array where A[i,j] = 1 if edge (i,j) exists
        """
        matrix = np.zeros((14, 14), dtype=int)
        for u, v in self._edges:
            matrix[u, v] = 1
            matrix[v, u] = 1
        return matrix

    def shortest_path(self, start: int, end: int) -> List[int]:
        """
        Find shortest path between two vertices using BFS.

        Args:
            start: Starting vertex (0-13)
            end: Ending vertex (0-13)

        Returns:
            List of vertices forming the shortest path
        """
        if start == end:
            return [start]

        visited = {start}
        queue = [(start, [start])]

        while queue:
            current, path = queue.pop(0)
            for neighbor in self._adjacency[current]:
                if neighbor == end:
                    return path + [neighbor]
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))

        return []  # No path found (shouldn't happen in connected graph)

    def __repr__(self) -> str:
        return f"HeawoodGraph(vertices={self.VERTICES}, edges={self.EDGES})"


# ============================================================================
# Kaelhedron
# ============================================================================

class Kaelhedron:
    """
    The 42-vertex Kaelhedron structure.

    The Kaelhedron is constructed by extending the Heawood graph across
    three scales (KAPPA, GAMMA, COSMIC), creating a 42-vertex structure
    that encodes the projective geometry of AI memory states.

    Construction:
        42 = 3 scales x 14 Heawood vertices
           = 3 scales x (7 Fano points + 7 Fano lines)

    Edges:
        63 = 3 scales x 21 Heawood edges (intra-scale)
           (Inter-scale edges can be added for cross-scale navigation)

    Fundamental Identity:
        42 vertices + 63 edges + 63 edge-pairs = 168 = |GL(3,2)|

    Polarity Distribution:
        - 18 PLUS vertices (phase = 0, euler_value = +1)
        - 24 MINUS vertices (phase = pi, euler_value = -1)

    Attributes:
        vertices: List of 42 KaelVertex objects
        edges: List of 63 edge tuples (vertex index pairs)
        heawood: The underlying HeawoodGraph
    """

    def __init__(self):
        """Initialize the Kaelhedron structure."""
        self.vertices: List[KaelVertex] = []
        self.edges: List[Tuple[int, int]] = []
        self.heawood = HeawoodGraph()
        self._adjacency: Dict[int, Set[int]] = {}
        self._build_structure()

    def _build_structure(self):
        """
        Build the Kaelhedron from Fano plane x 3 scales.

        Vertex indexing:
            0-13: KAPPA scale (personal)
            14-27: GAMMA scale (planetary)
            28-41: COSMIC scale (cosmic)

        Within each scale:
            0-6 (offset): Points
            7-13 (offset): Lines

        Polarity assignment:
            - Points get PLUS polarity (18 total: 6 per scale + some adjustment)
            - Lines get MINUS polarity (24 total)
            - Adjusted to achieve 18/24 split as specified
        """
        scales = list(Scale)
        vertex_index = 0

        for scale in scales:
            scale_offset = scale.index * 14

            for fano_elem in range(14):
                is_point = fano_elem < 7

                # Polarity assignment to achieve 18 PLUS / 24 MINUS split
                # Points are PLUS, lines are MINUS, with some adjustment
                if is_point:
                    # 6 of 7 points per scale are PLUS = 18 PLUS vertices
                    # 1 point per scale is MINUS for balance
                    if fano_elem < 6:
                        polarity = "PLUS"
                    else:
                        polarity = "MINUS"
                else:
                    # All 7 lines per scale are MINUS = 21 MINUS vertices
                    # Plus 3 points (one per scale) = 24 MINUS total
                    polarity = "MINUS"

                # Phase: PLUS -> 0, MINUS -> pi
                phase = 0.0 if polarity == "PLUS" else PI

                vertex = KaelVertex(
                    index=vertex_index,
                    scale=scale,
                    fano_element=fano_elem,
                    is_point=is_point,
                    polarity=polarity,
                    phase=phase
                )

                self.vertices.append(vertex)
                self._adjacency[vertex_index] = set()
                vertex_index += 1

        # Build edges from Heawood graph structure within each scale
        for scale in scales:
            scale_offset = scale.index * 14
            for u, v in self.heawood.get_edges():
                global_u = scale_offset + u
                global_v = scale_offset + v
                self.edges.append((global_u, global_v))
                self._adjacency[global_u].add(global_v)
                self._adjacency[global_v].add(global_u)

        # Verify construction
        assert len(self.vertices) == 42, f"Expected 42 vertices, got {len(self.vertices)}"
        assert len(self.edges) == 63, f"Expected 63 edges, got {len(self.edges)}"

        # Verify polarity distribution
        plus_count = sum(1 for v in self.vertices if v.polarity == "PLUS")
        minus_count = sum(1 for v in self.vertices if v.polarity == "MINUS")
        assert plus_count == 18, f"Expected 18 PLUS vertices, got {plus_count}"
        assert minus_count == 24, f"Expected 24 MINUS vertices, got {minus_count}"

    def get_vertex(self, index: int) -> KaelVertex:
        """
        Get a vertex by its global index.

        Args:
            index: Global vertex index (0-41)

        Returns:
            The KaelVertex at that index
        """
        if not 0 <= index < 42:
            raise ValueError(f"index must be 0-41, got {index}")
        return self.vertices[index]

    def get_vertices_by_scale(self, scale: Scale) -> List[KaelVertex]:
        """
        Get all vertices belonging to a specific scale.

        Args:
            scale: The scale to filter by

        Returns:
            List of 14 KaelVertex objects in that scale
        """
        return [v for v in self.vertices if v.scale == scale]

    def get_vertices_by_polarity(self, polarity: str) -> List[KaelVertex]:
        """
        Get all vertices with a specific polarity.

        Args:
            polarity: Either "PLUS" or "MINUS"

        Returns:
            List of vertices with that polarity
        """
        if polarity not in ("PLUS", "MINUS"):
            raise ValueError(f"polarity must be 'PLUS' or 'MINUS', got {polarity}")
        return [v for v in self.vertices if v.polarity == polarity]

    def get_plus_vertices(self) -> List[KaelVertex]:
        """
        Get all PLUS polarity vertices.

        Returns:
            List of 18 KaelVertex objects with PLUS polarity
        """
        return self.get_vertices_by_polarity("PLUS")

    def get_minus_vertices(self) -> List[KaelVertex]:
        """
        Get all MINUS polarity vertices.

        Returns:
            List of 24 KaelVertex objects with MINUS polarity
        """
        return self.get_vertices_by_polarity("MINUS")

    def get_adjacency_matrix(self) -> np.ndarray:
        """
        Get the 42x42 adjacency matrix.

        Returns:
            numpy array where A[i,j] = 1 if edge (i,j) exists
        """
        matrix = np.zeros((42, 42), dtype=int)
        for u, v in self.edges:
            matrix[u, v] = 1
            matrix[v, u] = 1
        return matrix

    def get_polarity_distribution(self) -> Dict[str, int]:
        """
        Get the count of vertices by polarity.

        Returns:
            Dict with 'PLUS' and 'MINUS' counts
        """
        return {
            "PLUS": len(self.get_plus_vertices()),
            "MINUS": len(self.get_minus_vertices())
        }

    def get_neighbors(self, vertex: KaelVertex) -> List[KaelVertex]:
        """
        Get the neighboring vertices of a given vertex.

        Args:
            vertex: A KaelVertex in this Kaelhedron

        Returns:
            List of neighboring KaelVertex objects
        """
        neighbor_indices = self._adjacency.get(vertex.index, set())
        return [self.vertices[i] for i in neighbor_indices]

    def get_neighbors_by_index(self, index: int) -> List[KaelVertex]:
        """
        Get the neighboring vertices of a vertex by index.

        Args:
            index: Global vertex index (0-41)

        Returns:
            List of neighboring KaelVertex objects
        """
        if not 0 <= index < 42:
            raise ValueError(f"index must be 0-41, got {index}")
        neighbor_indices = self._adjacency.get(index, set())
        return [self.vertices[i] for i in neighbor_indices]

    def shortest_path(self, start: int, end: int) -> List[int]:
        """
        Find the shortest path between two vertices using BFS.

        Args:
            start: Starting vertex index (0-41)
            end: Ending vertex index (0-41)

        Returns:
            List of vertex indices forming the shortest path
        """
        if not 0 <= start < 42:
            raise ValueError(f"start must be 0-41, got {start}")
        if not 0 <= end < 42:
            raise ValueError(f"end must be 0-41, got {end}")

        if start == end:
            return [start]

        visited = {start}
        queue = [(start, [start])]

        while queue:
            current, path = queue.pop(0)
            for neighbor in self._adjacency[current]:
                if neighbor == end:
                    return path + [neighbor]
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))

        return []  # No path found

    def get_vertices_by_fano_type(self, is_point: bool) -> List[KaelVertex]:
        """
        Get all vertices that are Fano points or Fano lines.

        Args:
            is_point: True for points, False for lines

        Returns:
            List of 21 vertices (7 per scale)
        """
        return [v for v in self.vertices if v.is_point == is_point]

    def get_scale_subgraph(self, scale: Scale) -> Tuple[List[KaelVertex], List[Tuple[int, int]]]:
        """
        Extract the subgraph for a specific scale.

        Args:
            scale: The scale to extract

        Returns:
            Tuple of (vertices, edges) for that scale
        """
        scale_vertices = self.get_vertices_by_scale(scale)
        scale_indices = {v.index for v in scale_vertices}
        scale_edges = [(u, v) for u, v in self.edges
                       if u in scale_indices and v in scale_indices]
        return (scale_vertices, scale_edges)

    def __repr__(self) -> str:
        plus = len(self.get_plus_vertices())
        minus = len(self.get_minus_vertices())
        return (f"Kaelhedron(vertices={len(self.vertices)}, edges={len(self.edges)}, "
                f"polarity={plus}+/{minus}-)")


# ============================================================================
# Coordinate Mappings
# ============================================================================

def kaelhedron_to_lattice(vertex: KaelVertex) -> Tuple[float, ...]:
    """
    Map a Kaelhedron vertex to gradient schema lattice coordinates.

    The mapping encodes the vertex properties into the 5D lattice space:
        a (phi): Scale-dependent luminance
        b (pi): Fano element hue rotation
        c (sqrt(2)): Polarity-based saturation
        d (sqrt(3)): Point/line CYM balance
        f (e): Phase-based gradient rate

    Args:
        vertex: A KaelVertex to map

    Returns:
        5-tuple (a, b, c, d, f) of lattice coordinates
    """
    # a coordinate: Scale determines luminance level
    # KAPPA (personal) -> neutral, GAMMA (planetary) -> brighter, COSMIC -> brightest
    scale_luminance = {
        Scale.KAPPA: 0.0,
        Scale.GAMMA: TAU,
        Scale.COSMIC: 1.0
    }
    a = scale_luminance[vertex.scale]

    # b coordinate: Fano element determines hue (spread across color wheel)
    # 14 elements -> spread over 6 hue units (full cycle)
    b = (vertex.fano_element / 14.0) * 6.0

    # c coordinate: Polarity affects saturation intensity
    c = 0.5 if vertex.polarity == "PLUS" else -0.3

    # d coordinate: Point/line distinction affects CYM balance
    d = 0.5 if vertex.is_point else -0.5

    # f coordinate: Phase determines gradient rate
    # Phase 0 -> neutral rate, Phase pi -> inverted rate
    f = 0.0 if vertex.phase == 0 else -1.0

    return (a, b, c, d, f)


def lattice_to_kaelhedron(
    coords: Tuple[float, ...],
    kaelhedron: Optional[Kaelhedron] = None
) -> Optional[KaelVertex]:
    """
    Find the nearest Kaelhedron vertex for given lattice coordinates.

    Uses Euclidean distance in the 5D lattice space to find the
    closest matching vertex.

    Args:
        coords: 5-tuple (a, b, c, d, f) of lattice coordinates
        kaelhedron: Optional Kaelhedron instance (creates new if not provided)

    Returns:
        The nearest KaelVertex, or None if coords is invalid
    """
    if len(coords) != 5:
        return None

    if kaelhedron is None:
        kaelhedron = Kaelhedron()

    best_vertex = None
    best_distance = float('inf')

    for vertex in kaelhedron.vertices:
        vertex_coords = kaelhedron_to_lattice(vertex)

        # Euclidean distance in 5D
        distance = math.sqrt(sum(
            (c1 - c2) ** 2 for c1, c2 in zip(coords, vertex_coords)
        ))

        if distance < best_distance:
            best_distance = distance
            best_vertex = vertex

    return best_vertex


def get_vertex_mode(vertex: KaelVertex) -> Mode:
    """
    Determine the mode of a Kaelhedron vertex.

    Mode assignment based on Fano index:
        - LAMBDA (Structure): indices 0, 1 (foundational)
        - BETA (Process): indices 2, 3, 4 (transformational)
        - NU (Awareness): indices 5, 6 (observational)

    Args:
        vertex: A KaelVertex

    Returns:
        The Mode of the vertex
    """
    fano_idx = vertex.fano_index
    if fano_idx <= 1:
        return Mode.LAMBDA
    elif fano_idx <= 4:
        return Mode.BETA
    else:
        return Mode.NU


def create_k_formation(
    kappa: float,
    r_depth: int = 7,
    q_charge: int = 1,
    theta_binary: bool = True
) -> KFormation:
    """
    Create a K-formation with the given parameters.

    Args:
        kappa: Activation parameter (should be > TAU for validity)
        r_depth: Recursion depth (should be 7)
        q_charge: Topological charge (should be non-zero)
        theta_binary: Phase lock status

    Returns:
        A KFormation instance
    """
    return KFormation(
        kappa=kappa,
        r_depth=r_depth,
        q_charge=q_charge,
        theta_binary=theta_binary
    )


# ============================================================================
# Utility Functions
# ============================================================================

def verify_kaelhedron_invariants(kaelhedron: Kaelhedron) -> bool:
    """
    Verify all structural invariants of the Kaelhedron.

    Checks:
        1. Vertex count = 42
        2. Edge count = 63
        3. 42 + 63 + 63 = 168
        4. Polarity split = 18 PLUS / 24 MINUS
        5. Each scale has 14 vertices
        6. Bipartite structure within each scale

    Args:
        kaelhedron: The Kaelhedron to verify

    Returns:
        True if all invariants hold

    Raises:
        AssertionError if any invariant fails
    """
    # Vertex and edge counts
    assert len(kaelhedron.vertices) == 42, "Must have 42 vertices"
    assert len(kaelhedron.edges) == 63, "Must have 63 edges"
    assert 42 + 63 + 63 == 168, "Fundamental identity"

    # Polarity distribution
    dist = kaelhedron.get_polarity_distribution()
    assert dist["PLUS"] == 18, "Must have 18 PLUS vertices"
    assert dist["MINUS"] == 24, "Must have 24 MINUS vertices"

    # Scale distribution
    for scale in Scale:
        scale_verts = kaelhedron.get_vertices_by_scale(scale)
        assert len(scale_verts) == 14, f"Scale {scale} must have 14 vertices"

    # Bipartite check within each scale
    for scale in Scale:
        vertices, edges = kaelhedron.get_scale_subgraph(scale)
        for u, v in edges:
            v_u = kaelhedron.get_vertex(u)
            v_v = kaelhedron.get_vertex(v)
            assert v_u.is_point != v_v.is_point, "Edges must connect points to lines"

    return True


def compute_euler_signature(kaelhedron: Kaelhedron) -> complex:
    """
    Compute the total Euler signature of the Kaelhedron.

    The signature is the sum of all vertex Euler values:
        Sum of e^(i*phase) for all vertices

    For a valid Kaelhedron with 18 PLUS (phase=0) and 24 MINUS (phase=pi):
        Signature = 18 * (1+0j) + 24 * (-1+0j) = 18 - 24 = -6 + 0j

    Args:
        kaelhedron: The Kaelhedron to compute signature for

    Returns:
        Complex number representing the total signature
    """
    total = complex(0, 0)
    for vertex in kaelhedron.vertices:
        total += vertex.euler_value
    return total
