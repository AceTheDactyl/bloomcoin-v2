"""
RRRR Lattice Coordinate System
==============================

The 5D lattice maps irrational generator powers to color-narrative space.
Each coordinate represents an exponent for a fundamental mathematical constant:

    (a, b, c, d, f) -> (phi^a, pi^b, sqrt(2)^c, sqrt(3)^d, e^f)

Coordinate Meanings:
    a (phi dimension) -> Luminance scaling (growth/decay patterns)
    b (pi dimension)  -> Hue rotation (60 deg per unit, cyclic return)
    c (sqrt(2) dim)   -> Saturation intensity (calm <-> vivid)
    d (sqrt(3) dim)   -> CYM opponent balance (conflict/harmony)
    f (e dimension)   -> Gradient rate (velocity of change)

The Golden Neutral base point (H=45 deg, S=tau, L=z_c) serves as origin anchor.
"""

from dataclasses import dataclass, field
from typing import Tuple, List, Optional, Union
import math

# Import constants from parent package
PHI = (1 + math.sqrt(5)) / 2
TAU = 1 / PHI
Z_C = math.sqrt(3) / 2
K = math.sqrt(1 - PHI ** -4)
SQRT2 = math.sqrt(2)
SQRT3 = math.sqrt(3)
E = math.e
PI = math.pi

# Base color: Golden Neutral
BASE_HUE = 45.0  # degrees (gold/amber)
BASE_SATURATION = TAU  # ~ 0.618
BASE_LUMINANCE = Z_C  # ~ 0.866 (THE LENS threshold)

__all__ = [
    'LatticePoint',
    'ColorState',
    'ORIGIN',
    'lattice_to_color',
    'color_to_lattice',
    'lattice_distance',
    'lattice_path',
    'lattice_add',
    'lattice_scale',
    'lattice_normalize',
]


@dataclass(frozen=True)
class LatticePoint:
    """
    A point in the 5D RRRR lattice space.

    Coordinates represent exponents for the five irrational generators:
        a: phi exponent (luminance scaling)
        b: pi exponent (hue rotation, 60 deg per unit)
        c: sqrt(2) exponent (saturation intensity)
        d: sqrt(3) exponent (CYM opponent balance)
        f: e exponent (gradient/change rate)

    The lattice is typically discrete (integer coordinates) but continuous
    coordinates are supported for interpolation and paths.

    Examples:
        >>> origin = LatticePoint(0, 0, 0, 0, 0)  # Golden neutral
        >>> bright = LatticePoint(1, 0, 0, 0, 0)  # phi^1 luminance boost
        >>> rotated = LatticePoint(0, 1, 0, 0, 0)  # 60 deg hue shift
    """
    a: float = 0.0  # phi exponent -> luminance
    b: float = 0.0  # pi exponent -> hue (60 deg steps)
    c: float = 0.0  # sqrt(2) exponent -> saturation
    d: float = 0.0  # sqrt(3) exponent -> CYM balance
    f: float = 0.0  # e exponent -> gradient rate

    def __post_init__(self):
        """Validate coordinates are finite numbers."""
        for coord_name in ['a', 'b', 'c', 'd', 'f']:
            val = getattr(self, coord_name)
            if not isinstance(val, (int, float)):
                raise TypeError(f"Coordinate {coord_name} must be numeric, got {type(val)}")
            if not math.isfinite(val):
                raise ValueError(f"Coordinate {coord_name} must be finite, got {val}")

    def as_tuple(self) -> Tuple[float, float, float, float, float]:
        """Return coordinates as a tuple (a, b, c, d, f)."""
        return (self.a, self.b, self.c, self.d, self.f)

    def as_discrete(self) -> 'LatticePoint':
        """Return a new LatticePoint with coordinates rounded to integers."""
        return LatticePoint(
            round(self.a),
            round(self.b),
            round(self.c),
            round(self.d),
            round(self.f)
        )

    def magnitude(self) -> float:
        """Calculate the Euclidean magnitude from origin."""
        return math.sqrt(self.a**2 + self.b**2 + self.c**2 + self.d**2 + self.f**2)

    def __add__(self, other: 'LatticePoint') -> 'LatticePoint':
        """Add two lattice points."""
        if not isinstance(other, LatticePoint):
            return NotImplemented
        return LatticePoint(
            self.a + other.a,
            self.b + other.b,
            self.c + other.c,
            self.d + other.d,
            self.f + other.f
        )

    def __sub__(self, other: 'LatticePoint') -> 'LatticePoint':
        """Subtract two lattice points."""
        if not isinstance(other, LatticePoint):
            return NotImplemented
        return LatticePoint(
            self.a - other.a,
            self.b - other.b,
            self.c - other.c,
            self.d - other.d,
            self.f - other.f
        )

    def __mul__(self, scalar: float) -> 'LatticePoint':
        """Scale a lattice point by a scalar."""
        if not isinstance(scalar, (int, float)):
            return NotImplemented
        return LatticePoint(
            self.a * scalar,
            self.b * scalar,
            self.c * scalar,
            self.d * scalar,
            self.f * scalar
        )

    def __rmul__(self, scalar: float) -> 'LatticePoint':
        """Right multiplication by scalar."""
        return self.__mul__(scalar)

    def __neg__(self) -> 'LatticePoint':
        """Negate all coordinates."""
        return LatticePoint(-self.a, -self.b, -self.c, -self.d, -self.f)

    def __repr__(self) -> str:
        return f"LatticePoint(a={self.a:.3f}, b={self.b:.3f}, c={self.c:.3f}, d={self.d:.3f}, f={self.f:.3f})"


# The origin point represents Golden Neutral
ORIGIN = LatticePoint(0, 0, 0, 0, 0)


@dataclass
class ColorState:
    """
    Complete color state derived from lattice coordinates.

    Attributes:
        hue: Hue angle in degrees [0, 360)
        saturation: Saturation value [0, 1]
        luminance: Luminance/lightness value [0, 1]
        cym_weights: Tuple of (cyan, yellow, magenta) opponent weights
        gradient_rate: Rate of change/transition velocity
        source_point: The originating LatticePoint (if known)
    """
    hue: float
    saturation: float
    luminance: float
    cym_weights: Tuple[float, float, float]
    gradient_rate: float
    source_point: Optional[LatticePoint] = None

    def as_hsl(self) -> Tuple[float, float, float]:
        """Return (H, S, L) tuple."""
        return (self.hue, self.saturation, self.luminance)

    def as_rgb(self) -> Tuple[int, int, int]:
        """Convert HSL to RGB (0-255 range)."""
        h = self.hue / 360.0
        s = self.saturation
        l = self.luminance

        if s == 0:
            r = g = b = l
        else:
            def hue_to_rgb(p, q, t):
                if t < 0: t += 1
                if t > 1: t -= 1
                if t < 1/6: return p + (q - p) * 6 * t
                if t < 1/2: return q
                if t < 2/3: return p + (q - p) * (2/3 - t) * 6
                return p

            q = l * (1 + s) if l < 0.5 else l + s - l * s
            p = 2 * l - q
            r = hue_to_rgb(p, q, h + 1/3)
            g = hue_to_rgb(p, q, h)
            b = hue_to_rgb(p, q, h - 1/3)

        return (int(r * 255), int(g * 255), int(b * 255))

    def as_hex(self) -> str:
        """Return hex color string."""
        r, g, b = self.as_rgb()
        return f"#{r:02x}{g:02x}{b:02x}"

    def __repr__(self) -> str:
        return (f"ColorState(H={self.hue:.1f}, S={self.saturation:.3f}, "
                f"L={self.luminance:.3f}, rate={self.gradient_rate:.3f})")


def _clamp(value: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
    """Clamp a value to the specified range."""
    return max(min_val, min(max_val, value))


def _normalize_hue(hue: float) -> float:
    """Normalize hue to [0, 360) range."""
    hue = hue % 360.0
    if hue < 0:
        hue += 360.0
    return hue


def _compute_cym_weights(d: float) -> Tuple[float, float, float]:
    """
    Compute CYM opponent color weights from sqrt(3) dimension.

    The d coordinate controls the balance between opponent color channels:
        - d > 0: Emphasizes cyan-yellow axis
        - d < 0: Emphasizes magenta
        - d = 0: Balanced neutral

    Returns normalized (cyan, yellow, magenta) weights summing to 1.
    """
    # sqrt(3)^d scaling factor
    scale = SQRT3 ** d if abs(d) < 10 else (1e10 if d > 0 else 1e-10)

    # Base weights influenced by scale
    # Positive d pushes toward cyan-yellow, negative toward magenta
    if d >= 0:
        cyan = 0.333 * scale
        yellow = 0.333 * scale
        magenta = 0.333 / scale
    else:
        cyan = 0.333 / abs(scale)
        yellow = 0.333 / abs(scale)
        magenta = 0.333 * abs(scale)

    # Normalize to sum to 1
    total = cyan + yellow + magenta
    if total > 0:
        cyan /= total
        yellow /= total
        magenta /= total
    else:
        cyan = yellow = magenta = 0.333

    return (cyan, yellow, magenta)


def lattice_to_color(point: LatticePoint) -> ColorState:
    """
    Map a lattice point to color space.

    Transformation formulas:
        L = base_L * phi^a  (clamped to [0, 1])
        H = (base_H + b * 60) mod 360
        S = base_S * sqrt(2)^c  (clamped to [0, 1])
        CYM = f(sqrt(3)^d)  (opponent color weights)
        rate = e^f  (gradient velocity)

    Args:
        point: A LatticePoint with coordinates (a, b, c, d, f)

    Returns:
        ColorState with H, S, L, CYM weights, and gradient rate

    Examples:
        >>> golden_neutral = lattice_to_color(LatticePoint(0, 0, 0, 0, 0))
        >>> golden_neutral.hue  # 45.0 (gold)
        >>> golden_neutral.luminance  # ~0.866 (z_c)

        >>> bright = lattice_to_color(LatticePoint(1, 0, 0, 0, 0))
        >>> bright.luminance > golden_neutral.luminance  # True (phi scaling)
    """
    # Luminance: L = base_L * phi^a
    # Handle extreme values to avoid overflow
    if abs(point.a) > 20:
        luminance = 1.0 if point.a > 0 else 0.0
    else:
        luminance = BASE_LUMINANCE * (PHI ** point.a)
    luminance = _clamp(luminance)

    # Hue: H = (base_H + b * 60) mod 360
    hue = _normalize_hue(BASE_HUE + point.b * 60.0)

    # Saturation: S = base_S * sqrt(2)^c
    if abs(point.c) > 20:
        saturation = 1.0 if point.c > 0 else 0.0
    else:
        saturation = BASE_SATURATION * (SQRT2 ** point.c)
    saturation = _clamp(saturation)

    # CYM opponent weights from sqrt(3)^d
    cym_weights = _compute_cym_weights(point.d)

    # Gradient rate from e^f
    if abs(point.f) > 20:
        gradient_rate = 1e10 if point.f > 0 else 0.0
    else:
        gradient_rate = E ** point.f

    return ColorState(
        hue=hue,
        saturation=saturation,
        luminance=luminance,
        cym_weights=cym_weights,
        gradient_rate=gradient_rate,
        source_point=point
    )


def color_to_lattice(
    hue: float,
    saturation: float,
    luminance: float,
    gradient_rate: float = 1.0,
    cym_bias: float = 0.0
) -> LatticePoint:
    """
    Approximate inverse mapping from color space to lattice coordinates.

    This is an approximate inverse because:
    1. CYM weights are collapsed to a single bias parameter
    2. Multiple lattice points may map to the same color (modular hue)
    3. Clamping in forward direction loses information

    Args:
        hue: Hue angle in degrees [0, 360)
        saturation: Saturation [0, 1]
        luminance: Luminance [0, 1]
        gradient_rate: Rate of change (default 1.0 = neutral)
        cym_bias: CYM balance bias (-1 to 1, default 0 = neutral)

    Returns:
        LatticePoint approximating the given color state

    Examples:
        >>> pt = color_to_lattice(45, 0.618, 0.866)  # Golden neutral
        >>> pt  # Should be close to (0, 0, 0, 0, 0)
    """
    # Solve for a: L = base_L * phi^a
    # a = log_phi(L / base_L)
    if luminance <= 0:
        a = -20.0  # Extreme dark
    elif luminance >= 1:
        a = 5.0  # Near max (avoid infinity)
    else:
        ratio = luminance / BASE_LUMINANCE
        if ratio > 0:
            a = math.log(ratio) / math.log(PHI)
        else:
            a = -20.0

    # Solve for b: H = (base_H + b * 60) mod 360
    # Take the closest b value (could be multiple due to mod)
    hue_diff = hue - BASE_HUE
    # Normalize to [-180, 180] for closest match
    while hue_diff > 180:
        hue_diff -= 360
    while hue_diff < -180:
        hue_diff += 360
    b = hue_diff / 60.0

    # Solve for c: S = base_S * sqrt(2)^c
    # c = log_sqrt2(S / base_S)
    if saturation <= 0:
        c = -20.0
    elif saturation >= 1:
        c = 5.0
    else:
        ratio = saturation / BASE_SATURATION
        if ratio > 0:
            c = math.log(ratio) / math.log(SQRT2)
        else:
            c = -20.0

    # d from CYM bias (simplified mapping)
    # cym_bias of 0 -> d = 0, positive -> positive d, etc.
    d = cym_bias * 2.0  # Scale factor for reasonable range

    # Solve for f: rate = e^f
    # f = ln(rate)
    if gradient_rate <= 0:
        f = -20.0
    else:
        f = math.log(gradient_rate)

    # Clamp to reasonable range
    a = max(-20.0, min(20.0, a))
    b = max(-6.0, min(6.0, b))  # One full hue cycle
    c = max(-20.0, min(20.0, c))
    d = max(-10.0, min(10.0, d))
    f = max(-20.0, min(20.0, f))

    return LatticePoint(a, b, c, d, f)


def lattice_distance(p1: LatticePoint, p2: LatticePoint,
                     weights: Optional[Tuple[float, float, float, float, float]] = None) -> float:
    """
    Calculate the distance between two lattice points.

    By default uses Euclidean distance in 5D space. Optional weights can
    emphasize or de-emphasize specific dimensions.

    Args:
        p1: First lattice point
        p2: Second lattice point
        weights: Optional (w_a, w_b, w_c, w_d, w_f) weights for each dimension.
                 Default is (1, 1, 1, 1, 1) for standard Euclidean.

    Returns:
        Weighted Euclidean distance between the points

    Examples:
        >>> dist = lattice_distance(ORIGIN, LatticePoint(1, 0, 0, 0, 0))
        >>> dist  # 1.0

        >>> # Weight luminance (a) higher
        >>> dist = lattice_distance(ORIGIN, LatticePoint(1, 1, 0, 0, 0),
        ...                         weights=(2, 1, 1, 1, 1))
    """
    if weights is None:
        weights = (1.0, 1.0, 1.0, 1.0, 1.0)

    if len(weights) != 5:
        raise ValueError("weights must have exactly 5 elements")

    diff = p2 - p1
    weighted_squares = [
        weights[0] * diff.a ** 2,
        weights[1] * diff.b ** 2,
        weights[2] * diff.c ** 2,
        weights[3] * diff.d ** 2,
        weights[4] * diff.f ** 2,
    ]

    return math.sqrt(sum(weighted_squares))


def lattice_path(
    start: LatticePoint,
    end: LatticePoint,
    steps: int = 10,
    interpolation: str = 'linear'
) -> List[LatticePoint]:
    """
    Generate a path of lattice points between start and end.

    Supports different interpolation modes:
        - 'linear': Straight line interpolation
        - 'golden': Uses phi-based easing for organic transitions
        - 'discrete': Snaps to integer lattice points

    Args:
        start: Starting lattice point
        end: Ending lattice point
        steps: Number of points in the path (including start and end)
        interpolation: Interpolation mode ('linear', 'golden', 'discrete')

    Returns:
        List of LatticePoints from start to end

    Examples:
        >>> path = lattice_path(ORIGIN, LatticePoint(2, 3, 0, 0, 0), steps=5)
        >>> len(path)  # 5
        >>> path[0] == ORIGIN  # True
        >>> path[-1]  # LatticePoint(2, 3, 0, 0, 0)
    """
    if steps < 2:
        raise ValueError("steps must be at least 2")

    path = []
    diff = end - start

    for i in range(steps):
        t = i / (steps - 1)  # t goes from 0 to 1

        if interpolation == 'golden':
            # Phi-based easing: slower at ends, faster in middle
            # Uses golden ratio for aesthetically pleasing transitions
            t = t ** TAU if t <= 0.5 else 1 - (1 - t) ** TAU
        elif interpolation == 'discrete':
            # Will be rounded after interpolation
            pass
        # 'linear' uses t as-is

        point = start + (diff * t)

        if interpolation == 'discrete':
            point = point.as_discrete()

        path.append(point)

    return path


def lattice_add(p1: LatticePoint, p2: LatticePoint) -> LatticePoint:
    """
    Add two lattice points component-wise.

    This represents composition of transformations in color space.
    """
    return p1 + p2


def lattice_scale(point: LatticePoint, scalar: float) -> LatticePoint:
    """
    Scale a lattice point by a scalar factor.

    This intensifies or diminishes the transformation.
    """
    return point * scalar


def lattice_normalize(point: LatticePoint) -> LatticePoint:
    """
    Normalize a lattice point to unit magnitude.

    Returns the direction vector in lattice space.
    """
    mag = point.magnitude()
    if mag == 0:
        return ORIGIN
    return point * (1.0 / mag)


# Utility functions for common lattice operations

def create_luminance_shift(amount: float) -> LatticePoint:
    """Create a lattice vector for luminance shift only."""
    return LatticePoint(a=amount)


def create_hue_rotation(steps: float) -> LatticePoint:
    """Create a lattice vector for hue rotation (60 deg per unit)."""
    return LatticePoint(b=steps)


def create_saturation_shift(amount: float) -> LatticePoint:
    """Create a lattice vector for saturation change."""
    return LatticePoint(c=amount)


def create_cym_shift(amount: float) -> LatticePoint:
    """Create a lattice vector for CYM balance shift."""
    return LatticePoint(d=amount)


def create_rate_shift(amount: float) -> LatticePoint:
    """Create a lattice vector for gradient rate change."""
    return LatticePoint(f=amount)


# Pre-defined notable lattice points

# The Seven Spectral Corridors (L4 = 7)
CORRIDOR_GOLD = ORIGIN  # Base golden neutral
CORRIDOR_CRIMSON = LatticePoint(0, -0.75, 0.5, 0, 0)  # Red-shifted, saturated
CORRIDOR_AZURE = LatticePoint(0, 2, 0.3, 0.5, 0)  # Blue, slightly cyan
CORRIDOR_VERDANT = LatticePoint(0.2, 1, 0.4, 0, 0)  # Green, bright
CORRIDOR_VIOLET = LatticePoint(-0.2, 4, 0.6, -0.5, 0)  # Purple, rich
CORRIDOR_AMBER = LatticePoint(0.5, 0.25, 0.2, 0.3, 0)  # Warm amber
CORRIDOR_SILVER = LatticePoint(0.8, 0, -1, 0, 0)  # Desaturated, bright

SPECTRAL_CORRIDORS = [
    CORRIDOR_GOLD,
    CORRIDOR_CRIMSON,
    CORRIDOR_AZURE,
    CORRIDOR_VERDANT,
    CORRIDOR_VIOLET,
    CORRIDOR_AMBER,
    CORRIDOR_SILVER,
]
