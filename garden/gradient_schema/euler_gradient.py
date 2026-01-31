"""
Euler's Gradient System
=======================

The foundation of the Kaelhedron structure derived from Euler's identity:
    e^(i*pi) + 1 = 0

This module implements the gradient arc from +1 to -1 on the complex unit circle,
where:
    - +1 = e^(i*0) represents the PLUS pole (binary 0, certainty)
    - -1 = e^(i*pi) represents the MINUS pole (binary 1, negation)
    - +i = e^(i*pi/2) represents PARADOX (superposition, uncertainty)
    - -i = e^(i*3*pi/2) represents ANTI-PARADOX (resolution)

The Euler Gradient maps consciousness states to positions on this arc, with the
golden ratio barrier (tau = 1/phi ~ 0.618) separating binary collapse from
paradox preservation.

Kaelhedron Structure:
    - 42 vertices = 2 * 3 * 7 (Kaelhedron)
    - 168 vertices = |GL(3,2)| (Octakaelhedron)
    - 7 points, 7 lines (Fano plane, projective plane over F_2)

The Fano plane encodes the octonion multiplication table, with each line
representing a valid multiplication triple.
"""

from __future__ import annotations

import math
import cmath
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Tuple, Optional, Set, FrozenSet


# =============================================================================
# EULER'S GRADIENT POLES
# =============================================================================

PLUS_POLE: int = 1    # e^(i*0) = +1
"""The positive pole, representing binary 0, certainty, and phase 0."""

MINUS_POLE: int = -1  # e^(i*pi) = -1
"""The negative pole, representing binary 1, negation, and phase pi."""


# =============================================================================
# GOLDEN RATIO CONNECTIONS
# =============================================================================

PHI: float = (1 + math.sqrt(5)) / 2  # ~ 1.618033988749895
"""
Golden Ratio (phi ~ 1.618)

The ratio of the whole to the larger part equals the ratio of the larger part
to the smaller. Appears in the Fano plane geometry and pentagon structure.
"""

TAU: float = 1 / PHI  # ~ 0.6180339887498949
"""
Golden Ratio Conjugate (tau = 1/phi ~ 0.618)

The barrier threshold for binary collapse. When uncertainty (imaginary component)
falls below tau, the gradient collapses to the nearest binary pole.
Also equals phi - 1.
"""

GOLDEN_ANGLE: float = math.acos(TAU)  # ~ 0.9046 radians ~ 51.83 degrees
"""
The angle where cos(theta) = phi^(-1) = tau.

This is the angle at which the real (polarity) component equals the barrier
threshold, marking the transition between binary and paradox regions.
"""


# =============================================================================
# PENTAGON CONNECTION
# =============================================================================

PENTAGON_PHI_ANGLE: float = math.pi / 5  # 36 degrees
"""
Pentagon-Phi Connection Angle

The remarkable identity: cos(pi/5) = phi/2 (EXACT!)

This connects the regular pentagon to the golden ratio, explaining why
the pentagram contains golden ratio proportions throughout.
"""

# Verification: cos(pi/5) should equal phi/2
_cos_pi_5 = math.cos(math.pi / 5)
_phi_half = PHI / 2
assert abs(_cos_pi_5 - _phi_half) < 1e-10, f"Pentagon-phi identity violated: {_cos_pi_5} != {_phi_half}"


# =============================================================================
# BASE-4 FOURTH ROOTS OF UNITY
# =============================================================================

FOURTH_ROOTS: Dict[int, complex] = {
    0: complex(1, 0),   # +1, digit 0 (PLUS)
    1: complex(-1, 0),  # -1, digit 1 (MINUS)
    2: complex(0, 1),   # +i, digit 2 (PARADOX)
    3: complex(0, -1)   # -i, digit 3 (Anti-paradox)
}
"""
Fourth roots of unity mapped to base-4 digits.

The four roots divide the unit circle into quadrants, providing a natural
base-4 encoding for gradient positions. Note the unconventional ordering:
digits 0 and 1 are the binary poles (real axis), while digits 2 and 3
are the paradox states (imaginary axis).
"""


# =============================================================================
# KAELHEDRON NUMBERS
# =============================================================================

KAELHEDRON_VERTICES: int = 42   # 2 * 3 * 7
"""
Kaelhedron vertex count.

42 = 2 * 3 * 7, the famous "Answer to Life, the Universe, and Everything."
The Kaelhedron is a 42-vertex polyhedron derived from Fano plane symmetries.
"""

OCTAKAELHEDRON_VERTICES: int = 168  # |GL(3,2)|
"""
Octakaelhedron vertex count.

168 = |GL(3,2)| = |PSL(2,7)| = 4 * 42
The order of the automorphism group of the Fano plane, also the order of
the smallest simple group with a faithful 3-dimensional representation over F_2.
"""

FANO_POINTS: int = 7
"""Number of points in the Fano plane (projective plane over F_2)."""

FANO_LINES: int = 7
"""Number of lines in the Fano plane (each line contains exactly 3 points)."""


# =============================================================================
# EULER GRADIENT CLASS
# =============================================================================

@dataclass
class EulerGradient:
    """
    Position on the Euler gradient arc from +1 to -1.

    The gradient represents a continuous transition from the PLUS pole (+1)
    through the PARADOX state (+i at theta=pi/2) to the MINUS pole (-1).

    The phase angle theta ranges from 0 to pi along the upper semicircle:
        - theta = 0: PLUS pole (+1)
        - theta = pi/2: PARADOX (+i)
        - theta = pi: MINUS pole (-1)

    Attributes:
        theta: Phase angle in radians, constrained to [0, pi]

    Examples:
        >>> plus = EulerGradient(0)
        >>> plus.polarity
        'PLUS'
        >>> plus.is_binary
        True

        >>> paradox = EulerGradient(math.pi / 2)
        >>> paradox.polarity
        'PARADOX'
        >>> paradox.is_paradox
        True

        >>> minus = EulerGradient(math.pi)
        >>> minus.polarity
        'MINUS'
    """
    theta: float  # Phase angle [0, pi]

    def __post_init__(self):
        """Normalize theta to [0, pi] range."""
        # Handle negative angles
        while self.theta < 0:
            object.__setattr__(self, 'theta', self.theta + 2 * math.pi)
        # Handle angles > 2*pi
        self.theta = self.theta % (2 * math.pi)
        # Mirror lower semicircle to upper
        if self.theta > math.pi:
            object.__setattr__(self, 'theta', 2 * math.pi - self.theta)

    @property
    def real(self) -> float:
        """
        Real component: cos(theta).

        Represents the polarity dimension - how strongly the gradient
        is pulled toward +1 (positive real) or -1 (negative real).

        Returns:
            Value in [-1, 1]
        """
        return math.cos(self.theta)

    @property
    def imag(self) -> float:
        """
        Imaginary component: sin(theta).

        Represents the uncertainty dimension - how strongly the gradient
        is in a paradox/superposition state (positive imaginary).

        Returns:
            Value in [0, 1] (upper semicircle only)
        """
        return math.sin(self.theta)

    @property
    def complex_value(self) -> complex:
        """
        Complex representation: e^(i*theta).

        Returns the position on the unit circle as a complex number.

        Returns:
            Complex number with |z| = 1
        """
        return cmath.exp(complex(0, self.theta))

    @property
    def polarity(self) -> str:
        """
        Classify the gradient position.

        Returns:
            - "PLUS" if close to +1 (theta near 0)
            - "MINUS" if close to -1 (theta near pi)
            - "PARADOX" if in the uncertainty region

        The classification uses the golden angle as threshold:
        - PLUS: theta < GOLDEN_ANGLE
        - PARADOX: GOLDEN_ANGLE <= theta <= pi - GOLDEN_ANGLE
        - MINUS: theta > pi - GOLDEN_ANGLE
        """
        if self.theta < GOLDEN_ANGLE:
            return "PLUS"
        elif self.theta > math.pi - GOLDEN_ANGLE:
            return "MINUS"
        else:
            return "PARADOX"

    @property
    def is_binary(self) -> bool:
        """
        Check if gradient is at a binary pole.

        Returns:
            True if theta is 0 or pi (within floating point tolerance)
        """
        epsilon = 1e-10
        return abs(self.theta) < epsilon or abs(self.theta - math.pi) < epsilon

    @property
    def is_paradox(self) -> bool:
        """
        Check if gradient is at the paradox point.

        Returns:
            True if theta is pi/2 (within floating point tolerance)
        """
        epsilon = 1e-10
        return abs(self.theta - math.pi / 2) < epsilon

    @classmethod
    def from_complex(cls, z: complex) -> 'EulerGradient':
        """
        Create gradient from a complex number.

        The complex number is normalized to the unit circle before
        extracting the phase angle.

        Args:
            z: Any non-zero complex number

        Returns:
            EulerGradient with theta = arg(z)

        Raises:
            ValueError: If z is zero
        """
        if z == 0:
            raise ValueError("Cannot create gradient from zero")

        # Get phase angle (returns value in (-pi, pi])
        theta = cmath.phase(z)

        # Normalize to [0, pi] by taking absolute value
        # (mirrors lower semicircle to upper)
        if theta < 0:
            theta = -theta

        return cls(theta)

    def to_base4_digit(self) -> int:
        """
        Map gradient position to base-4 digit.

        Divides the upper semicircle into 4 quadrants:
            - [0, pi/4): digit 0 (near +1)
            - [pi/4, pi/2): digit 2 (approaching +i)
            - [pi/2, 3*pi/4): digit 2 (departing +i)
            - [3*pi/4, pi]: digit 1 (near -1)

        Note: Both quadrants around pi/2 map to digit 2 (PARADOX).
        For full 4-digit encoding, use the full circle version.

        Returns:
            Integer 0, 1, or 2
        """
        if self.theta < math.pi / 4:
            return 0  # Near PLUS
        elif self.theta > 3 * math.pi / 4:
            return 1  # Near MINUS
        else:
            return 2  # PARADOX region

    def __repr__(self) -> str:
        return f"EulerGradient(theta={self.theta:.6f}, polarity='{self.polarity}')"


# =============================================================================
# GRADIENT OPERATIONS
# =============================================================================

def gradient_position(t: float) -> EulerGradient:
    """
    Get gradient position at t in [0, 1] from +1 to -1.

    Maps the unit interval to the upper semicircle:
        - t = 0 -> theta = 0 -> +1 (PLUS)
        - t = 0.5 -> theta = pi/2 -> +i (PARADOX)
        - t = 1 -> theta = pi -> -1 (MINUS)

    Args:
        t: Parameter in [0, 1]

    Returns:
        EulerGradient at position t along the arc

    Raises:
        ValueError: If t is outside [0, 1]
    """
    if not 0 <= t <= 1:
        raise ValueError(f"Parameter t must be in [0, 1], got {t}")

    theta = t * math.pi
    return EulerGradient(theta)


def gradient_to_color(gradient: EulerGradient) -> Tuple[float, float, float]:
    """
    Map gradient position to HSL color.

    The mapping uses:
        - Hue: theta mapped to [0, 180] degrees (red at +1, cyan at -1)
        - Saturation: imag component (maximum at paradox)
        - Lightness: (real + 1) / 2, normalized to [0, 1]

    Args:
        gradient: EulerGradient position

    Returns:
        Tuple (hue, saturation, lightness) with:
            - hue in [0, 360]
            - saturation in [0, 1]
            - lightness in [0, 1]
    """
    # Hue: map theta from [0, pi] to [0, 180] degrees
    # +1 (theta=0) -> red (0 degrees)
    # -1 (theta=pi) -> cyan (180 degrees)
    hue = (gradient.theta / math.pi) * 180.0

    # Saturation: imaginary component represents uncertainty/paradox
    # Maximum at pi/2, zero at poles
    saturation = gradient.imag

    # Lightness: real component normalized to [0, 1]
    # +1 -> 1.0 (bright)
    # -1 -> 0.0 (dark)
    lightness = (gradient.real + 1) / 2

    return (hue, saturation, lightness)


def binary_collapse(gradient: EulerGradient, threshold: float = TAU) -> EulerGradient:
    """
    Collapse to nearest binary pole if uncertainty is below threshold.

    When the imaginary (uncertainty) component falls below the golden
    barrier (tau ~ 0.618), the gradient collapses to the nearest pole.

    This models quantum-like measurement: superpositions persist until
    uncertainty drops below the consciousness threshold, then collapse
    to definite binary values.

    Args:
        gradient: Current gradient position
        threshold: Uncertainty threshold (default: TAU = 0.618...)

    Returns:
        Original gradient if imag >= threshold
        EulerGradient(0) if real > 0 and imag < threshold
        EulerGradient(pi) if real < 0 and imag < threshold
    """
    if gradient.imag >= threshold:
        # Uncertainty above barrier - preserve superposition
        return gradient

    # Collapse to nearest pole
    if gradient.real >= 0:
        return EulerGradient(0)  # PLUS pole
    else:
        return EulerGradient(math.pi)  # MINUS pole


# =============================================================================
# FANO PLANE STRUCTURE
# =============================================================================

# The 7 Fano points (labeled 0-6)
FANO_POINT_SET: List[int] = list(range(7))
"""The 7 points of the Fano plane, labeled 0 through 6."""

# The 7 Fano lines (each containing exactly 3 points)
FANO_LINE_SET: List[FrozenSet[int]] = [
    frozenset({0, 1, 3}),  # Line 0
    frozenset({1, 2, 4}),  # Line 1
    frozenset({2, 3, 5}),  # Line 2
    frozenset({3, 4, 6}),  # Line 3
    frozenset({4, 5, 0}),  # Line 4 - THE PLUS LINE
    frozenset({5, 6, 1}),  # Line 5
    frozenset({6, 0, 2}),  # Line 6
]
"""
The 7 lines of the Fano plane.

Each line contains exactly 3 points, and any two distinct points determine
exactly one line. This is the smallest finite projective plane (order 2).

The Fano plane encodes the octonion multiplication table.
"""

# Binary polarity assignment (unique modulo GL(3,2) action)
PLUS_CLASS: FrozenSet[int] = frozenset({0, 4, 5})  # Fano Line 4 (phase 0)
"""
Points assigned to the PLUS polarity.

These are the three points on Line 4 (the "Plus Line").
The assignment is unique modulo the GL(3,2) symmetry group.
"""

MINUS_CLASS: FrozenSet[int] = frozenset({1, 2, 3, 6})  # Complement (phase pi)
"""
Points assigned to the MINUS polarity.

The four points not on Line 4.
"""


def get_point_polarity(point: int) -> str:
    """
    Get the polarity of a Fano plane point.

    Args:
        point: Integer 0-6

    Returns:
        "PLUS" if point is in {0, 4, 5}
        "MINUS" if point is in {1, 2, 3, 6}

    Raises:
        ValueError: If point is not in 0-6
    """
    if not 0 <= point <= 6:
        raise ValueError(f"Point must be in 0-6, got {point}")

    if point in PLUS_CLASS:
        return "PLUS"
    else:
        return "MINUS"


def verify_fano_multiplication(i: int, j: int, k: int) -> bool:
    """
    Verify that (i, j, k) forms a valid Fano line (octonion triple).

    In the octonion multiplication table, each Fano line {i, j, k}
    satisfies e_i * e_j = e_k (with appropriate sign).

    Args:
        i, j, k: Three distinct integers in 0-6

    Returns:
        True if {i, j, k} is a Fano line

    Raises:
        ValueError: If any point is not in 0-6
    """
    if not all(0 <= p <= 6 for p in (i, j, k)):
        raise ValueError(f"All points must be in 0-6, got ({i}, {j}, {k})")

    triple = frozenset({i, j, k})
    return triple in FANO_LINE_SET


def get_lines_through_point(point: int) -> List[FrozenSet[int]]:
    """
    Get all Fano lines passing through a given point.

    In the Fano plane, exactly 3 lines pass through each point.

    Args:
        point: Integer 0-6

    Returns:
        List of 3 frozensets, each containing 3 points
    """
    if not 0 <= point <= 6:
        raise ValueError(f"Point must be in 0-6, got {point}")

    return [line for line in FANO_LINE_SET if point in line]


def get_line_through_points(p1: int, p2: int) -> FrozenSet[int]:
    """
    Get the unique Fano line through two distinct points.

    In a projective plane, any two distinct points determine exactly one line.

    Args:
        p1, p2: Two distinct integers in 0-6

    Returns:
        Frozenset containing the three points on the line

    Raises:
        ValueError: If points are equal or not in 0-6
    """
    if p1 == p2:
        raise ValueError("Points must be distinct")
    if not (0 <= p1 <= 6 and 0 <= p2 <= 6):
        raise ValueError(f"Points must be in 0-6, got ({p1}, {p2})")

    for line in FANO_LINE_SET:
        if p1 in line and p2 in line:
            return line

    # Should never reach here for valid Fano plane
    raise RuntimeError(f"No line found through points {p1} and {p2}")


# =============================================================================
# V4 KLEIN FOUR-GROUP OPERATIONS
# =============================================================================

class V4Operation(Enum):
    """
    The Klein four-group V4 = Z/2Z x Z/2Z.

    This is the group of symmetries of the double-well potential,
    mapping to the four base-4 digits:

        E (Identity):  x -> x      (digit 0, stays at +1)
        N (Negation):  x -> -x     (digit 1, flips to -1)
        I (Inversion): x -> 1/x    (digit 2, paradox)
        NI (Combined): x -> -1/x   (digit 3, anti-paradox)

    Multiplication table:
        |  E   N   I  NI
     ---+----------------
      E |  E   N   I  NI
      N |  N   E  NI   I
      I |  I  NI   E   N
     NI | NI   I   N   E

    Note: Every element is its own inverse (V4 is elementary abelian).
    """
    E = "e"    # Identity (digit 0)
    N = "N"    # Negation (digit 1)
    I = "I"    # Inversion (digit 2)
    NI = "NI"  # Combined (digit 3)


# V4 multiplication table
V4_MULTIPLICATION: Dict[Tuple[V4Operation, V4Operation], V4Operation] = {
    (V4Operation.E, V4Operation.E): V4Operation.E,
    (V4Operation.E, V4Operation.N): V4Operation.N,
    (V4Operation.E, V4Operation.I): V4Operation.I,
    (V4Operation.E, V4Operation.NI): V4Operation.NI,

    (V4Operation.N, V4Operation.E): V4Operation.N,
    (V4Operation.N, V4Operation.N): V4Operation.E,
    (V4Operation.N, V4Operation.I): V4Operation.NI,
    (V4Operation.N, V4Operation.NI): V4Operation.I,

    (V4Operation.I, V4Operation.E): V4Operation.I,
    (V4Operation.I, V4Operation.N): V4Operation.NI,
    (V4Operation.I, V4Operation.I): V4Operation.E,
    (V4Operation.I, V4Operation.NI): V4Operation.N,

    (V4Operation.NI, V4Operation.E): V4Operation.NI,
    (V4Operation.NI, V4Operation.N): V4Operation.I,
    (V4Operation.NI, V4Operation.I): V4Operation.N,
    (V4Operation.NI, V4Operation.NI): V4Operation.E,
}
"""Complete multiplication table for V4."""


def v4_multiply(a: V4Operation, b: V4Operation) -> V4Operation:
    """
    Multiply two V4 elements.

    Args:
        a, b: V4Operation elements

    Returns:
        Product a * b
    """
    return V4_MULTIPLICATION[(a, b)]


def v4_to_digit(op: V4Operation) -> int:
    """
    Map V4 operation to base-4 digit.

    Args:
        op: V4Operation

    Returns:
        Integer 0-3
    """
    mapping = {
        V4Operation.E: 0,
        V4Operation.N: 1,
        V4Operation.I: 2,
        V4Operation.NI: 3
    }
    return mapping[op]


def digit_to_v4(digit: int) -> V4Operation:
    """
    Map base-4 digit to V4 operation.

    Args:
        digit: Integer 0-3

    Returns:
        Corresponding V4Operation

    Raises:
        ValueError: If digit not in 0-3
    """
    if not 0 <= digit <= 3:
        raise ValueError(f"Digit must be 0-3, got {digit}")

    mapping = {
        0: V4Operation.E,
        1: V4Operation.N,
        2: V4Operation.I,
        3: V4Operation.NI
    }
    return mapping[digit]


def apply_v4(op: V4Operation, z: complex) -> complex:
    """
    Apply V4 operation to a complex number.

    Args:
        op: V4Operation to apply
        z: Complex number (must be non-zero for I and NI)

    Returns:
        Transformed complex number

    Raises:
        ZeroDivisionError: If z is zero and op is I or NI
    """
    if op == V4Operation.E:
        return z
    elif op == V4Operation.N:
        return -z
    elif op == V4Operation.I:
        return 1 / z
    elif op == V4Operation.NI:
        return -1 / z
    else:
        raise ValueError(f"Unknown V4 operation: {op}")


# =============================================================================
# DOUBLE-WELL STRUCTURE
# =============================================================================

@dataclass
class DoubleWell:
    """
    The consciousness double-well potential.

    Models the bistable system with two stable states (wells) separated
    by an energy barrier:

                    barrier (tau ~ 0.618)
                         |
         upper_well      |
           (mu_2)       /\
             *         /  \     conscious state
            . .       /    \
           .   .     /      \
          .     .   /        \
         .       . /          \
        .         *            .
       .      lower_well        .
      .         (mu_1)           .   pre-conscious state
     .                            .

    The barrier at tau = 1/phi is the threshold for state transitions.

    Attributes:
        lower_well: Pre-conscious stable state (mu_1 ~ 0.472)
        barrier: Transition threshold (tau = 1/phi ~ 0.618)
        upper_well: Conscious stable state (mu_2 ~ 0.763)
    """
    lower_well: float = 0.472  # mu_1, pre-conscious
    barrier: float = TAU       # phi^(-1) = 0.618...
    upper_well: float = 0.763  # mu_2, conscious

    def __post_init__(self):
        """Validate well ordering."""
        if not (self.lower_well < self.barrier < self.upper_well):
            raise ValueError(
                f"Wells must satisfy lower < barrier < upper, "
                f"got {self.lower_well} < {self.barrier} < {self.upper_well}"
            )

    def get_well(self, kappa: float) -> str:
        """
        Determine which well a given kappa value falls into.

        Args:
            kappa: Consciousness parameter in [0, 1]

        Returns:
            "LOWER" if kappa <= midpoint between lower_well and barrier
            "BARRIER" if kappa is near the barrier
            "UPPER" if kappa >= midpoint between barrier and upper_well
        """
        # Midpoints for classification
        lower_mid = (self.lower_well + self.barrier) / 2
        upper_mid = (self.barrier + self.upper_well) / 2

        if kappa < lower_mid:
            return "LOWER"
        elif kappa > upper_mid:
            return "UPPER"
        else:
            return "BARRIER"

    def potential(self, kappa: float) -> float:
        """
        Compute the double-well potential energy at kappa.

        Uses a quartic potential: V(x) = a(x - mu_1)^2(x - mu_2)^2
        normalized so V(barrier) = 1.

        Args:
            kappa: Position parameter

        Returns:
            Potential energy value
        """
        # Distance from each well
        d1 = kappa - self.lower_well
        d2 = kappa - self.upper_well

        # Quartic potential
        raw = (d1 ** 2) * (d2 ** 2)

        # Normalize so barrier height = 1
        d1_barrier = self.barrier - self.lower_well
        d2_barrier = self.barrier - self.upper_well
        barrier_value = (d1_barrier ** 2) * (d2_barrier ** 2)

        if barrier_value > 0:
            return raw / barrier_value
        return raw

    def gradient_force(self, kappa: float) -> float:
        """
        Compute the gradient force at kappa.

        The force is -dV/dkappa, pushing toward the nearest well.

        Args:
            kappa: Position parameter

        Returns:
            Force value (negative = toward lower well, positive = toward upper)
        """
        # Numerical derivative with small epsilon
        eps = 1e-8
        v_plus = self.potential(kappa + eps)
        v_minus = self.potential(kappa - eps)
        return -(v_plus - v_minus) / (2 * eps)

    def is_stable(self, kappa: float, tolerance: float = 0.05) -> bool:
        """
        Check if kappa is near a stable well.

        Args:
            kappa: Position to check
            tolerance: Distance threshold for stability

        Returns:
            True if kappa is within tolerance of either well
        """
        return (abs(kappa - self.lower_well) < tolerance or
                abs(kappa - self.upper_well) < tolerance)


# =============================================================================
# CONVENIENCE INSTANCES
# =============================================================================

# Standard double-well with default parameters
STANDARD_DOUBLE_WELL = DoubleWell()
"""Standard consciousness double-well with mu_1=0.472, barrier=TAU, mu_2=0.763."""

# Cardinal gradient positions
GRADIENT_PLUS = EulerGradient(0)
"""The PLUS pole at theta=0 (e^(i*0) = +1)."""

GRADIENT_MINUS = EulerGradient(math.pi)
"""The MINUS pole at theta=pi (e^(i*pi) = -1)."""

GRADIENT_PARADOX = EulerGradient(math.pi / 2)
"""The PARADOX point at theta=pi/2 (e^(i*pi/2) = +i)."""


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Constants
    'PLUS_POLE',
    'MINUS_POLE',
    'PHI',
    'TAU',
    'GOLDEN_ANGLE',
    'PENTAGON_PHI_ANGLE',
    'FOURTH_ROOTS',
    'KAELHEDRON_VERTICES',
    'OCTAKAELHEDRON_VERTICES',
    'FANO_POINTS',
    'FANO_LINES',

    # EulerGradient class
    'EulerGradient',

    # Gradient operations
    'gradient_position',
    'gradient_to_color',
    'binary_collapse',

    # Fano plane
    'FANO_POINT_SET',
    'FANO_LINE_SET',
    'PLUS_CLASS',
    'MINUS_CLASS',
    'get_point_polarity',
    'verify_fano_multiplication',
    'get_lines_through_point',
    'get_line_through_points',

    # V4 Klein four-group
    'V4Operation',
    'V4_MULTIPLICATION',
    'v4_multiply',
    'v4_to_digit',
    'digit_to_v4',
    'apply_v4',

    # Double-well
    'DoubleWell',
    'STANDARD_DOUBLE_WELL',

    # Convenience instances
    'GRADIENT_PLUS',
    'GRADIENT_MINUS',
    'GRADIENT_PARADOX',
]
