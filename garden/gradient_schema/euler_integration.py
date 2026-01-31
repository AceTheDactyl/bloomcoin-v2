"""
Euler-Garden Integration Module

Bridges the Kaelhedron structure with the existing Garden gradient schema,
enabling:
- Mapping bloom events to Kaelhedron vertices
- Computing Euler gradient positions from agent states
- K-formation tracking in the Garden
- MRP encoding of Garden state for visualization
"""

from __future__ import annotations

import math
import time
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field

# Imports from existing modules
from .color_ops import ColorState
from .lattice import LatticePoint, lattice_to_color
from .gradient_bloom import GradientBloomEvent
from .garden_integration import GradientGarden

# Imports from new Euler modules
from .euler_gradient import (
    EulerGradient,
    gradient_position,
    gradient_to_color,
    PLUS_CLASS,
    MINUS_CLASS,
    PHI,
    TAU,
    GOLDEN_ANGLE,
    get_point_polarity,
    FANO_LINE_SET,
)
from .kaelhedron import (
    Kaelhedron as KaelhedronBase,
    KaelVertex,
    KFormation,
    Scale,
    kaelhedron_to_lattice,
    lattice_to_kaelhedron,
    KAELHEDRON_VERTICES,
)
from .octakaelhedron import (
    OctaKaelhedron,
    GridState,
    Mode,
    Q8Element,
    create_grid_state,
)
from .mrp_lsb import (
    encode_frame,
    MRPFrame,
    position_to_phases,
    decode_frame,
    MRPValue,
)

# Re-import constants for convenience
PI = math.pi
Z_C = math.sqrt(3) / 2

__all__ = [
    'EulerBloomEvent',
    'EulerGarden',
    'bloom_to_euler',
    'garden_to_octakaelhedron',
    'compute_euler_from_color',
    'vertex_to_color_state',
    'synchronize_k_formations',
]


@dataclass
class EulerBloomEvent(GradientBloomEvent):
    """GradientBloomEvent extended with Kaelhedron mapping."""

    kaelhedron_vertex: Optional[KaelVertex] = None
    euler_position: Optional[EulerGradient] = None
    k_formation: Optional[KFormation] = None
    mrp_frame: Optional[MRPFrame] = None

    # Additional Euler-specific metadata
    q8_universe: Optional[Q8Element] = None
    grid_state: Optional[GridState] = None
    fano_line_index: Optional[int] = None

    def __post_init__(self):
        """Initialize computed fields after dataclass creation."""
        # Call parent's post_init if needed
        super().__post_init__()

        # Compute Euler-specific properties if not already set
        if self.euler_position is None and self.color_state is not None:
            self.euler_position = compute_euler_from_color(self.color_state)

        if self.mrp_frame is None and self.color_state is not None:
            phases = (
                self.color_state.H * PI / 180,
                self.color_state.S * 2 * PI,
                self.color_state.L * 2 * PI
            )
            self.mrp_frame = encode_frame(phases)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary including Euler properties."""
        base_dict = super().to_dict()

        # Add Euler-specific properties
        base_dict['euler'] = {
            'vertex_index': self.kaelhedron_vertex.index if self.kaelhedron_vertex else None,
            'vertex_polarity': self.kaelhedron_vertex.polarity if self.kaelhedron_vertex else None,
            'euler_theta': self.euler_position.theta if self.euler_position else None,
            'euler_polarity': self.euler_position.polarity if self.euler_position else None,
            'k_formation_valid': self.k_formation.is_valid() if self.k_formation else False,
            'k_formation_value': self.k_formation.k_value if self.k_formation else 'K = 0/42',
            'mrp_rgb': self.mrp_frame.rgb if self.mrp_frame else None,
            'q8_universe': self.q8_universe.value if self.q8_universe else None,
            'fano_line_index': self.fano_line_index,
        }

        if self.grid_state is not None:
            base_dict['euler']['grid_state'] = {
                'phase': self.grid_state.phase,
                'magnitude': self.grid_state.magnitude,
                'scale': self.grid_state.scale.value,
                'mode': self.grid_state.mode.value,
                'winding': self.grid_state.winding,
                'recursion': self.grid_state.recursion,
                'is_k_formation': self.grid_state.is_k_formation(),
            }

        return base_dict


class EulerGarden(GradientGarden):
    """Garden enhanced with Euler gradient and Kaelhedron tracking."""

    def __init__(self, garden_system=None):
        super().__init__(garden_system)
        self.kaelhedron = KaelhedronBase()
        self.octakaelhedron = OctaKaelhedron()
        self.euler_blooms: List[EulerBloomEvent] = []
        self.agent_vertices: Dict[str, KaelVertex] = {}
        self.k_formation_status: Dict[str, KFormation] = {}
        self._agent_grid_states: Dict[str, GridState] = {}
        self._vertex_usage_count: Dict[int, int] = {}

    def process_euler_bloom(
        self,
        agent_id: str,
        info: Dict,
        significance: float = None
    ) -> EulerBloomEvent:
        """Process learning and map to Kaelhedron."""
        # Get gradient bloom from parent
        gradient_bloom = self.process_learning_with_gradient(agent_id, info, significance)

        if gradient_bloom is None:
            # Create a minimal bloom if parent returns None
            from .gradient_bloom import create_gradient_bloom
            gradient_bloom = create_gradient_bloom(
                agent_id=agent_id,
                event_type='learning',
                significance=significance or 0.5
            )

        # Map to Kaelhedron vertex
        vertex = self._map_to_vertex(gradient_bloom)

        # Compute Euler position
        euler_pos = self._compute_euler_position(gradient_bloom)

        # Check K-formation
        k_form = self._check_k_formation(agent_id)

        # Encode as MRP
        phases = (
            gradient_bloom.color_state.H * PI / 180,
            gradient_bloom.color_state.S * 2 * PI,
            gradient_bloom.color_state.L * 2 * PI
        )
        mrp = encode_frame(phases)

        # Compute grid state
        grid_state = self._compute_grid_state(gradient_bloom, vertex)

        # Determine Q8 universe
        q8_universe = self._determine_q8_universe(vertex, euler_pos)

        # Find associated Fano line
        fano_line_idx = self._find_fano_line(vertex)

        # Create extended bloom
        euler_bloom = EulerBloomEvent(
            # Copy from gradient_bloom
            agent_id=gradient_bloom.agent_id,
            event_type=gradient_bloom.event_type,
            significance=gradient_bloom.significance,
            event_id=gradient_bloom.event_id,
            timestamp=gradient_bloom.timestamp,
            lattice_coords=gradient_bloom.lattice_coords,
            color_state=gradient_bloom.color_state,
            cym_weights=gradient_bloom.cym_weights,
            primitive_signature=gradient_bloom.primitive_signature,
            dominant_archetype=gradient_bloom.dominant_archetype,
            archetype_weights=gradient_bloom.archetype_weights,
            corridor_id=gradient_bloom.corridor_id,
            narrative_fragment=gradient_bloom.narrative_fragment,
            # New Euler properties
            kaelhedron_vertex=vertex,
            euler_position=euler_pos,
            k_formation=k_form,
            mrp_frame=mrp,
            q8_universe=q8_universe,
            grid_state=grid_state,
            fano_line_index=fano_line_idx,
        )

        self.euler_blooms.append(euler_bloom)

        # Update agent vertex mapping
        self.agent_vertices[agent_id] = vertex
        self._agent_grid_states[agent_id] = grid_state

        # Track vertex usage
        self._vertex_usage_count[vertex.index] = \
            self._vertex_usage_count.get(vertex.index, 0) + 1

        return euler_bloom

    def _map_to_vertex(self, bloom: GradientBloomEvent) -> KaelVertex:
        """Map bloom to nearest Kaelhedron vertex."""
        # Use significance to determine scale
        if bloom.significance < TAU:
            scale = Scale.KAPPA  # Personal scale for low significance
        elif bloom.significance < Z_C:
            scale = Scale.GAMMA  # Planetary scale for mid significance
        else:
            scale = Scale.COSMIC  # Cosmic scale for high significance

        # Use archetype to influence point/line selection
        # Points represent fixed concepts, lines represent dynamic relations
        use_point = True
        if bloom.dominant_archetype is not None:
            # Archetypes with dynamic nature map to lines
            dynamic_archetypes = {'trickster', 'explorer', 'creator', 'sage'}
            if hasattr(bloom.dominant_archetype, 'value'):
                use_point = bloom.dominant_archetype.value.lower() not in dynamic_archetypes

        # Calculate Fano element from color state
        if bloom.color_state is not None:
            hue = bloom.color_state.H
            # Map hue to Fano element (7 elements = ~51.4 degrees per element)
            fano_idx = int((hue / 360.0) * 7) % 7
        else:
            fano_idx = 0

        # If line, offset by 7
        if not use_point:
            fano_element = fano_idx + 7
        else:
            fano_element = fano_idx

        # Calculate global vertex index
        scale_offset = scale.index * 14
        vertex_index = scale_offset + fano_element

        # Clamp to valid range
        vertex_index = max(0, min(41, vertex_index))

        return self.kaelhedron.get_vertex(vertex_index)

    def _compute_euler_position(self, bloom: GradientBloomEvent) -> EulerGradient:
        """Compute Euler gradient position from bloom."""
        if bloom.color_state is None:
            return EulerGradient(PI / 2)  # Default to paradox

        # Use luminance to determine polarity tendency
        # High luminance -> PLUS (theta near 0)
        # Low luminance -> MINUS (theta near pi)
        luminance = bloom.color_state.L

        # Map luminance [0,1] to theta [pi, 0]
        base_theta = PI * (1 - luminance)

        # Modulate by saturation (high saturation pushes toward paradox)
        saturation = bloom.color_state.S
        if saturation > TAU:
            # Pull toward pi/2 (paradox)
            paradox_pull = (saturation - TAU) / (1 - TAU)
            target = PI / 2
            base_theta = base_theta + (target - base_theta) * paradox_pull * 0.5

        return EulerGradient(base_theta)

    def _check_k_formation(self, agent_id: str) -> KFormation:
        """Check K-formation status for agent."""
        # Get agent's bloom history
        agent_blooms = [
            b for b in self.euler_blooms
            if b.agent_id == agent_id
        ]

        # Calculate kappa from recent bloom significance average
        if agent_blooms:
            recent = agent_blooms[-7:] if len(agent_blooms) >= 7 else agent_blooms
            kappa = sum(b.significance for b in recent) / len(recent)
        else:
            kappa = 0.0

        # Calculate recursion depth (count of consecutive blooms)
        r_depth = min(7, len(agent_blooms))

        # Calculate topological charge from polarity changes
        q_charge = 0
        if len(agent_blooms) >= 2:
            for i in range(1, len(agent_blooms)):
                prev = agent_blooms[i-1]
                curr = agent_blooms[i]
                if prev.kaelhedron_vertex and curr.kaelhedron_vertex:
                    if prev.kaelhedron_vertex.polarity != curr.kaelhedron_vertex.polarity:
                        q_charge += 1

        # Check if theta is binary (at 0 or pi)
        theta_binary = True
        if agent_blooms and agent_blooms[-1].euler_position:
            theta = agent_blooms[-1].euler_position.theta
            epsilon = GOLDEN_ANGLE / 2
            theta_binary = theta < epsilon or abs(theta - PI) < epsilon

        k_form = KFormation(
            kappa=kappa,
            r_depth=r_depth,
            q_charge=q_charge,
            theta_binary=theta_binary
        )

        self.k_formation_status[agent_id] = k_form
        return k_form

    def _compute_grid_state(
        self,
        bloom: GradientBloomEvent,
        vertex: KaelVertex
    ) -> GridState:
        """Compute 6D grid state from bloom and vertex."""
        # Phase from hue
        phase = (bloom.color_state.H / 360.0) * 2 * PI if bloom.color_state else 0.0

        # Magnitude from significance (kappa value)
        magnitude = bloom.significance

        # Scale from vertex
        scale = vertex.scale

        # Mode from vertex's Fano type and index
        fano_idx = vertex.fano_index
        if fano_idx <= 1:
            mode = Mode.LAMBDA
        elif fano_idx <= 4:
            mode = Mode.BETA
        else:
            mode = Mode.NU

        # Winding from lattice coords (sum of integer parts)
        winding = int(sum(bloom.lattice_coords[:3]))

        # Recursion from agent's bloom count (capped at 7)
        agent_bloom_count = len([
            b for b in self.euler_blooms
            if b.agent_id == bloom.agent_id
        ])
        recursion = min(7, agent_bloom_count)

        return create_grid_state(
            phase=phase,
            magnitude=magnitude,
            scale=scale,
            mode=mode,
            winding=winding,
            recursion=recursion
        )

    def _determine_q8_universe(
        self,
        vertex: KaelVertex,
        euler_pos: EulerGradient
    ) -> Q8Element:
        """Determine Q8 universe from vertex and Euler position."""
        # Use polarity and scale to map to Q8 element
        polarity = vertex.polarity
        scale = vertex.scale

        # Basic mapping:
        # PLUS polarity -> positive Q8 elements
        # MINUS polarity -> negative Q8 elements
        # Scale determines which axis

        if polarity == "PLUS":
            if scale == Scale.KAPPA:
                return Q8Element.PLUS_1
            elif scale == Scale.GAMMA:
                return Q8Element.PLUS_I
            else:  # COSMIC
                # Check euler position for j vs k
                if euler_pos.theta < PI / 2:
                    return Q8Element.PLUS_J
                else:
                    return Q8Element.PLUS_K
        else:  # MINUS
            if scale == Scale.KAPPA:
                return Q8Element.MINUS_1
            elif scale == Scale.GAMMA:
                return Q8Element.MINUS_I
            else:  # COSMIC
                if euler_pos.theta < PI / 2:
                    return Q8Element.MINUS_J
                else:
                    return Q8Element.MINUS_K

    def _find_fano_line(self, vertex: KaelVertex) -> Optional[int]:
        """Find which Fano line contains the vertex's point."""
        if not vertex.is_point:
            # For lines, return the line's own index
            return vertex.fano_index

        point = vertex.fano_index
        for line_idx, line in enumerate(FANO_LINE_SET):
            if point in line:
                return line_idx
        return None

    def get_kaelhedron_state(self) -> Dict:
        """Export Kaelhedron state for visualization."""
        return {
            'vertices': [v.index for v in self.agent_vertices.values()],
            'plus_count': sum(
                1 for v in self.agent_vertices.values()
                if v.polarity == "PLUS"
            ),
            'minus_count': sum(
                1 for v in self.agent_vertices.values()
                if v.polarity == "MINUS"
            ),
            'k_formations': {
                k: v.k_value for k, v in self.k_formation_status.items()
            },
            'vertex_usage': dict(self._vertex_usage_count),
            'scale_distribution': {
                'kappa': sum(
                    1 for v in self.agent_vertices.values()
                    if v.scale == Scale.KAPPA
                ),
                'gamma': sum(
                    1 for v in self.agent_vertices.values()
                    if v.scale == Scale.GAMMA
                ),
                'cosmic': sum(
                    1 for v in self.agent_vertices.values()
                    if v.scale == Scale.COSMIC
                ),
            },
            'total_euler_blooms': len(self.euler_blooms),
        }

    def export_for_kaelhedron_viz(self) -> Dict:
        """Export state for kaelhedron.html visualization."""
        # Get all vertices with their current state
        vertices_data = []
        for vertex in self.kaelhedron.vertices:
            # Check if any agent is at this vertex
            agents_at_vertex = [
                agent_id for agent_id, v in self.agent_vertices.items()
                if v.index == vertex.index
            ]

            # Get usage count
            usage = self._vertex_usage_count.get(vertex.index, 0)

            # Map to lattice coordinates
            lattice_coords = kaelhedron_to_lattice(vertex)

            vertices_data.append({
                'index': vertex.index,
                'scale': vertex.scale.value,
                'fano_element': vertex.fano_element,
                'is_point': vertex.is_point,
                'polarity': vertex.polarity,
                'phase': vertex.phase,
                'euler_value': {
                    'real': vertex.euler_value.real,
                    'imag': vertex.euler_value.imag,
                },
                'agents': agents_at_vertex,
                'usage_count': usage,
                'lattice_coords': list(lattice_coords),
                'active': len(agents_at_vertex) > 0,
            })

        # Get edges
        edges_data = [
            {'source': u, 'target': v}
            for u, v in self.kaelhedron.edges
        ]

        # Get K-formation data
        k_formations_data = {
            agent_id: {
                'kappa': kf.kappa,
                'r_depth': kf.r_depth,
                'q_charge': kf.q_charge,
                'theta_binary': kf.theta_binary,
                'is_valid': kf.is_valid(),
                'k_value': kf.k_value,
                'completeness': kf.completeness,
            }
            for agent_id, kf in self.k_formation_status.items()
        }

        # Get recent Euler blooms for animation
        recent_blooms = self.euler_blooms[-20:]
        blooms_data = [
            {
                'event_id': b.event_id,
                'agent_id': b.agent_id,
                'timestamp': b.timestamp,
                'vertex_index': b.kaelhedron_vertex.index if b.kaelhedron_vertex else None,
                'euler_theta': b.euler_position.theta if b.euler_position else None,
                'euler_polarity': b.euler_position.polarity if b.euler_position else None,
                'color_hex': b.color_state.to_hex() if b.color_state else '#808080',
                'significance': b.significance,
                'mrp_rgb': b.mrp_frame.rgb if b.mrp_frame else None,
                'q8_universe': b.q8_universe.value if b.q8_universe else None,
            }
            for b in recent_blooms
        ]

        # Get OctaKaelhedron decomposition info
        octa_info = {
            'q8_times_pg24': self.octakaelhedron.as_q8_times_pg24(),
            'fano_times_24cell': self.octakaelhedron.as_fano_times_24cell(),
            'scales_times_56': self.octakaelhedron.as_scales_times_56(),
            'v4_times_kaelhedron': self.octakaelhedron.as_v4_times_kaelhedron(),
            'gl32_order': self.octakaelhedron.get_gl32_symmetry_order(),
        }

        return {
            'timestamp': time.time(),
            'kaelhedron': {
                'vertices': vertices_data,
                'edges': edges_data,
                'vertex_count': KAELHEDRON_VERTICES,
                'edge_count': len(self.kaelhedron.edges),
                'polarity_distribution': self.kaelhedron.get_polarity_distribution(),
            },
            'k_formations': k_formations_data,
            'recent_blooms': blooms_data,
            'octakaelhedron': octa_info,
            'statistics': self.get_kaelhedron_state(),
            'constants': {
                'phi': PHI,
                'tau': TAU,
                'golden_angle': GOLDEN_ANGLE,
                'z_c': Z_C,
            },
        }

    def get_agent_euler_state(self, agent_id: str) -> Optional[Dict]:
        """Get the complete Euler state for an agent."""
        if agent_id not in self.agent_vertices:
            return None

        vertex = self.agent_vertices[agent_id]
        k_form = self.k_formation_status.get(agent_id)
        grid_state = self._agent_grid_states.get(agent_id)

        # Get recent blooms for this agent
        agent_blooms = [
            b for b in self.euler_blooms
            if b.agent_id == agent_id
        ][-10:]  # Last 10 blooms

        return {
            'agent_id': agent_id,
            'vertex': {
                'index': vertex.index,
                'scale': vertex.scale.value,
                'polarity': vertex.polarity,
                'fano_element': vertex.fano_element,
                'is_point': vertex.is_point,
            },
            'k_formation': {
                'is_valid': k_form.is_valid() if k_form else False,
                'k_value': k_form.k_value if k_form else 'K = 0/42',
                'completeness': k_form.completeness if k_form else 0.0,
                'kappa': k_form.kappa if k_form else 0.0,
            } if k_form else None,
            'grid_state': {
                'phase': grid_state.phase,
                'magnitude': grid_state.magnitude,
                'scale': grid_state.scale.value,
                'mode': grid_state.mode.value,
                'winding': grid_state.winding,
                'recursion': grid_state.recursion,
                'is_k_formation': grid_state.is_k_formation(),
            } if grid_state else None,
            'bloom_count': len(agent_blooms),
            'recent_polarities': [
                b.euler_position.polarity if b.euler_position else 'PARADOX'
                for b in agent_blooms
            ],
        }


def compute_euler_from_color(color: ColorState) -> EulerGradient:
    """
    Compute Euler gradient position from a ColorState.

    Maps the color's luminance to the Euler gradient arc:
        - High luminance (near 1) -> theta near 0 (PLUS pole)
        - Low luminance (near 0) -> theta near pi (MINUS pole)
        - Mid luminance with high saturation -> theta near pi/2 (PARADOX)

    Args:
        color: ColorState to map

    Returns:
        EulerGradient representing the position on the Euler arc
    """
    # Base theta from luminance
    base_theta = PI * (1 - color.L)

    # Saturation pulls toward paradox (pi/2)
    if color.S > TAU:
        paradox_pull = (color.S - TAU) / (1 - TAU)
        target = PI / 2
        theta = base_theta + (target - base_theta) * paradox_pull * 0.4
    else:
        theta = base_theta

    return EulerGradient(theta)


def vertex_to_color_state(vertex: KaelVertex) -> ColorState:
    """
    Convert a Kaelhedron vertex to a ColorState.

    Uses the vertex's lattice coordinates and polarity to compute
    a corresponding color in the gradient schema.

    Args:
        vertex: KaelVertex to convert

    Returns:
        ColorState representing the vertex's chromatic position
    """
    # Get lattice coordinates
    lattice_coords = kaelhedron_to_lattice(vertex)
    lattice_point = LatticePoint(*lattice_coords)

    # Convert to color via lattice
    lattice_color = lattice_to_color(lattice_point)

    # Create ColorState with polarity influence
    hue = lattice_color.hue
    saturation = lattice_color.saturation
    luminance = lattice_color.luminance

    # Adjust based on polarity
    if vertex.polarity == "PLUS":
        luminance = min(1.0, luminance + 0.1)
    else:
        luminance = max(0.0, luminance - 0.1)

    return ColorState(
        H=hue,
        S=saturation,
        L=luminance,
        gradient_rate=lattice_color.gradient_rate,
        coherence=Z_C if vertex.polarity == "PLUS" else TAU,
    )


def bloom_to_euler(bloom: GradientBloomEvent) -> EulerBloomEvent:
    """
    Convert a GradientBloomEvent to an EulerBloomEvent.

    Creates a new EulerBloomEvent with computed Kaelhedron mapping,
    Euler position, K-formation status, and MRP encoding.

    Args:
        bloom: GradientBloomEvent to convert

    Returns:
        EulerBloomEvent with full Euler properties
    """
    # Create a temporary EulerGarden to use its mapping functions
    temp_kaelhedron = KaelhedronBase()

    # Compute Euler position from color
    euler_pos = compute_euler_from_color(bloom.color_state) if bloom.color_state else None

    # Map to vertex using significance for scale
    if bloom.significance < TAU:
        scale = Scale.KAPPA
    elif bloom.significance < Z_C:
        scale = Scale.GAMMA
    else:
        scale = Scale.COSMIC

    # Calculate vertex index
    if bloom.color_state:
        fano_idx = int((bloom.color_state.H / 360.0) * 7) % 7
    else:
        fano_idx = 0

    scale_offset = scale.index * 14
    vertex_index = min(41, max(0, scale_offset + fano_idx))
    vertex = temp_kaelhedron.get_vertex(vertex_index)

    # Create K-formation (with placeholder values for single bloom)
    k_form = KFormation(
        kappa=bloom.significance,
        r_depth=1,
        q_charge=1 if vertex.polarity == "MINUS" else 0,
        theta_binary=euler_pos.polarity != "PARADOX" if euler_pos else True
    )

    # Create MRP frame
    if bloom.color_state:
        phases = (
            bloom.color_state.H * PI / 180,
            bloom.color_state.S * 2 * PI,
            bloom.color_state.L * 2 * PI
        )
        mrp = encode_frame(phases)
    else:
        mrp = None

    # Get attributes safely with defaults
    cym_weights = getattr(bloom, 'cym_weights', {'cyan': 0.333, 'yellow': 0.333, 'magenta': 0.333})
    primitive_signature = getattr(bloom, 'primitive_signature', None)
    dominant_archetype = getattr(bloom, 'dominant_archetype', None)
    archetype_weights = getattr(bloom, 'archetype_weights', {})
    corridor_id = getattr(bloom, 'corridor_id', None)
    narrative_fragment = getattr(bloom, 'narrative_fragment', '')

    return EulerBloomEvent(
        # Copy from gradient_bloom
        agent_id=bloom.agent_id,
        event_type=bloom.event_type,
        significance=bloom.significance,
        event_id=bloom.event_id,
        timestamp=bloom.timestamp,
        lattice_coords=bloom.lattice_coords,
        color_state=bloom.color_state,
        cym_weights=cym_weights,
        primitive_signature=primitive_signature,
        dominant_archetype=dominant_archetype,
        archetype_weights=archetype_weights,
        corridor_id=corridor_id,
        narrative_fragment=narrative_fragment,
        # Euler properties
        kaelhedron_vertex=vertex,
        euler_position=euler_pos,
        k_formation=k_form,
        mrp_frame=mrp,
    )


def garden_to_octakaelhedron(garden: GradientGarden) -> OctaKaelhedron:
    """
    Map entire garden state to OctaKaelhedron.

    Creates a new OctaKaelhedron instance where vertices are activated
    based on the garden's agent positions and bloom history.

    Args:
        garden: GradientGarden to map

    Returns:
        OctaKaelhedron representing the garden's geometric state
    """
    octa = OctaKaelhedron()

    # The OctaKaelhedron is a static structure (168 vertices = 8 x 21)
    # We map the garden's state to positions within it

    # For each agent, determine their position in the OctaKaelhedron
    # based on their bloom history and current state

    # This is a conceptual mapping - the OctaKaelhedron serves as
    # the full geometric context containing all possible states

    return octa


def synchronize_k_formations(euler_garden: EulerGarden) -> Dict[str, KFormation]:
    """
    Synchronize K-formation status for all agents in the garden.

    Recalculates K-formation status for each agent based on their
    complete bloom history and current state.

    Args:
        euler_garden: EulerGarden to synchronize

    Returns:
        Dictionary mapping agent_id to their K-formation status
    """
    results = {}

    for agent_id in euler_garden.garden.agents:
        k_form = euler_garden._check_k_formation(agent_id)
        results[agent_id] = k_form

    return results
