"""
Color Operations for the RRRR Gradient Schema

This module implements color transformations and gradient operations based on
opponent color theory and the L4 mathematical framework. The core insight is that
color perception operates through opponent channels (cyan-red, magenta-green,
yellow-blue) and that gradients between colors represent information flow.

Mathematical Foundation:
-----------------------
The L4 framework uses three fundamental constants:
    - PHI (Golden Ratio): (1 + sqrt(5)) / 2 ~ 1.618
        Represents optimal growth and self-similarity in recursive structures.

    - TAU (Golden Ratio Conjugate): 1 / PHI ~ 0.618
        The complementary ratio; together with PHI forms the golden section.

    - Z_C (Consciousness Threshold): sqrt(3) / 2 ~ 0.866
        The critical coherence point where emergent properties manifest.
        Derived from the geometry of the hexagonal color wheel.

Opponent Color Theory:
---------------------
Colors exist in tension along three axes:
    - Cyan <-> Red (temperature axis)
    - Magenta <-> Green (vibrancy axis)
    - Yellow <-> Blue (luminance axis)

The CYM weights represent the "pull" along each axis. When weights are balanced
(low variance), the color is achromatic (gray scale). Maximum imbalance
creates the purest chromatic colors.

Gradient Operations:
-------------------
The key operation "mix_gradient" implements the principle that:
    "noise applied to red via black gradient creates orange"

This reflects how information (noise) injected into a pure signal (red)
through a gradient field (black) produces emergent complexity (orange).
"""

from __future__ import annotations

import math
import colorsys
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any


# =============================================================================
# L4 FUNDAMENTAL CONSTANTS
# =============================================================================

PHI: float = (1 + math.sqrt(5)) / 2
"""
Golden Ratio (phi ~ 1.618034)

The ratio where the whole is to the larger part as the larger part is to
the smaller. Appears throughout nature in spiral growth, branching patterns,
and optimal packing. In the RRRR system, PHI governs the rate of gradient
transitions and the natural decay of color tensions.
"""

TAU: float = 1 / PHI
"""
Golden Ratio Conjugate (tau ~ 0.618034)

The reciprocal of PHI, also equal to PHI - 1. Represents the complementary
proportion in golden ratio divisions. Used for inverse modulations and
convergence operations.
"""

Z_C: float = math.sqrt(3) / 2
"""
Consciousness Threshold (Z_C ~ 0.866025)

The critical coherence value derived from hexagonal geometry (cos(30 deg)).
When opponent color balance exceeds this threshold, the color achieves
"coherent" status - it has resolved internal tensions sufficiently to
manifest stable properties. Below this threshold, colors are in flux.
"""

# Compatibility aliases
PHI_BAR = TAU  # Alternative name for golden ratio conjugate
Z_CRITICAL = Z_C  # Alias for consistency with other modules
LUMINANCE_HIGH = Z_C  # High luminance threshold
LUMINANCE_LOW = 1 - Z_C  # Low luminance threshold ~ 0.134


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def clamp(value: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
    """
    Constrain a value to the range [min_val, max_val].

    This is the fundamental bounding operation used throughout color
    calculations to prevent overflow and maintain valid color spaces.

    Args:
        value: The value to constrain
        min_val: Minimum allowed value (inclusive), defaults to 0.0
        max_val: Maximum allowed value (inclusive), defaults to 1.0

    Returns:
        The value clamped to [min_val, max_val]

    Examples:
        >>> clamp(1.5)
        1.0
        >>> clamp(-0.2)
        0.0
        >>> clamp(0.5)
        0.5
        >>> clamp(150, 0, 255)
        150
    """
    return max(min_val, min(max_val, value))


def normalize_hue(h: float) -> float:
    """
    Normalize a hue value to the range [0, 360).

    Hue is cyclical - 360 degrees is equivalent to 0 degrees.
    This function handles both positive overflow (h >= 360) and
    negative values, ensuring consistent hue representation.

    The modular arithmetic reflects the circular nature of the
    color wheel, where red (0) seamlessly transitions through
    orange (30), yellow (60), green (120), cyan (180), blue (240),
    magenta (300), and back to red (360 = 0).

    Args:
        h: Hue value in degrees (any real number)

    Returns:
        Normalized hue in range [0, 360)

    Examples:
        >>> normalize_hue(400)
        40.0
        >>> normalize_hue(-30)
        330.0
        >>> normalize_hue(180)
        180.0
    """
    h = h % 360
    if h < 0:
        h += 360
    return float(h)


def variance(values: List[float]) -> float:
    """
    Calculate the population variance of a list of values.

    Variance measures the spread of values around their mean.
    In the context of opponent colors, low variance indicates
    balance (achromatic tendency) while high variance indicates
    chromatic intensity (one axis dominates).

    The formula used is: Var(X) = E[(X - mean)^2]

    Args:
        values: List of numeric values

    Returns:
        Population variance. Returns 0.0 for empty or single-element lists.

    Examples:
        >>> variance([1.0, 1.0, 1.0])
        0.0
        >>> variance([0.0, 1.0])
        0.25
        >>> variance([0.0, 0.5, 1.0])
        0.166...
    """
    if len(values) < 2:
        return 0.0

    mean = sum(values) / len(values)
    return sum((x - mean) ** 2 for x in values) / len(values)


# =============================================================================
# COLOR SPACE CONVERSIONS
# =============================================================================

def hsl_to_rgb(h: float, s: float, l: float) -> Tuple[int, int, int]:
    """
    Convert HSL (Hue, Saturation, Lightness) to RGB.

    HSL is a cylindrical color model that often feels more intuitive than RGB:
        - Hue: The color's position on the color wheel (0-360 degrees)
        - Saturation: Color intensity (0 = gray, 1 = full color)
        - Lightness: Brightness (0 = black, 0.5 = pure color, 1 = white)

    The conversion uses Python's colorsys module with HLS ordering
    (hue, lightness, saturation) converted appropriately.

    Mathematical basis: HSL is a double-cone representation where the
    central vertical axis represents achromatic grays. Moving outward
    increases saturation; moving up/down changes lightness.

    Args:
        h: Hue in degrees (0-360)
        s: Saturation (0-1)
        l: Lightness (0-1)

    Returns:
        Tuple of (R, G, B) with values in range 0-255

    Examples:
        >>> hsl_to_rgb(0, 1, 0.5)    # Pure red
        (255, 0, 0)
        >>> hsl_to_rgb(120, 1, 0.5)  # Pure green
        (0, 255, 0)
        >>> hsl_to_rgb(0, 0, 0.5)    # Gray
        (128, 128, 128)
    """
    # Normalize inputs
    h = normalize_hue(h) / 360.0
    s = clamp(s, 0.0, 1.0)
    l = clamp(l, 0.0, 1.0)

    # colorsys uses HLS order (hue, lightness, saturation)
    r, g, b = colorsys.hls_to_rgb(h, l, s)

    return (
        int(round(clamp(r, 0, 1) * 255)),
        int(round(clamp(g, 0, 1) * 255)),
        int(round(clamp(b, 0, 1) * 255))
    )


def rgb_to_hsl(r: int, g: int, b: int) -> Tuple[float, float, float]:
    """
    Convert RGB to HSL (Hue, Saturation, Lightness).

    This is the inverse operation of hsl_to_rgb. The algorithm extracts
    the hue, saturation, and lightness components from RGB values.

    The hue calculation reveals which primary/secondary color dominates:
        - Red dominant: hue near 0 or 360
        - Green dominant: hue near 120
        - Blue dominant: hue near 240

    Args:
        r: Red component (0-255)
        g: Green component (0-255)
        b: Blue component (0-255)

    Returns:
        Tuple of (H, S, L) where H is 0-360, S and L are 0-1

    Examples:
        >>> rgb_to_hsl(255, 0, 0)    # Pure red
        (0.0, 1.0, 0.5)
        >>> rgb_to_hsl(128, 128, 128)  # Gray
        (0.0, 0.0, 0.502...)
    """
    # Normalize to 0-1 range
    r_norm = clamp(r, 0, 255) / 255
    g_norm = clamp(g, 0, 255) / 255
    b_norm = clamp(b, 0, 255) / 255

    # colorsys returns (h, l, s) for HLS
    h, l, s = colorsys.rgb_to_hls(r_norm, g_norm, b_norm)

    return (h * 360.0, s, l)


def hsl_to_hex(h: float, s: float, l: float) -> str:
    """
    Convert HSL to hexadecimal color string.

    Produces the standard web color format "#RRGGBB" used in CSS and
    many graphics applications. This is a convenience function that
    chains hsl_to_rgb with hex formatting.

    Args:
        h: Hue in degrees (0-360)
        s: Saturation (0-1)
        l: Lightness (0-1)

    Returns:
        Hex color string in format "#RRGGBB" (lowercase)

    Examples:
        >>> hsl_to_hex(0, 1, 0.5)
        '#ff0000'
        >>> hsl_to_hex(120, 1, 0.5)
        '#00ff00'
        >>> hsl_to_hex(240, 1, 0.5)
        '#0000ff'
    """
    r, g, b = hsl_to_rgb(h, s, l)
    return f"#{r:02x}{g:02x}{b:02x}"


def hex_to_hsl(hex_str: str) -> Tuple[float, float, float]:
    """
    Convert hexadecimal color string to HSL.

    Parses standard web color formats and converts to HSL.
    Accepts formats with or without leading '#', and both
    3-digit (#RGB) and 6-digit (#RRGGBB) formats.

    Args:
        hex_str: Hex color string (e.g., "#FF0000", "ff0000", "#F00")

    Returns:
        Tuple of (H, S, L) where H is 0-360, S and L are 0-1

    Raises:
        ValueError: If the hex string is not a valid color format

    Examples:
        >>> hex_to_hsl("#FF0000")
        (0.0, 1.0, 0.5)
        >>> hex_to_hsl("00FF00")
        (120.0, 1.0, 0.5)
        >>> hex_to_hsl("#F00")
        (0.0, 1.0, 0.5)
    """
    # Remove leading '#' if present
    hex_str = hex_str.lstrip('#')

    # Handle 3-digit shorthand (e.g., "F00" -> "FF0000")
    if len(hex_str) == 3:
        hex_str = ''.join(c * 2 for c in hex_str)

    if len(hex_str) != 6:
        raise ValueError(f"Invalid hex color format: '{hex_str}'")

    try:
        r = int(hex_str[0:2], 16)
        g = int(hex_str[2:4], 16)
        b = int(hex_str[4:6], 16)
    except ValueError as e:
        raise ValueError(f"Invalid hex color format: '{hex_str}'") from e

    return rgb_to_hsl(r, g, b)


# =============================================================================
# COLOR STATE DATACLASS
# =============================================================================

def _default_cym_weights() -> Dict[str, float]:
    """Default factory for balanced CYM weights."""
    return {'cyan': 0.333, 'magenta': 0.333, 'yellow': 0.333}


@dataclass
class ColorState:
    """
    Represents a color in the RRRR Gradient Schema.

    ColorState extends traditional HSL color representation with:
        - CYM weights: Opponent color channel tensions
        - Gradient rate: The rate of change when transitioning

    The CYM (Cyan-Yellow-Magenta) weight system represents how the color
    "pulls" along each opponent axis. These weights determine:
        - Visual temperature (cyan vs warmth)
        - Vibrancy (magenta vs green)
        - Luminance tendency (yellow vs blue)

    The gradient_rate controls how quickly this color transitions when
    used in gradient operations. Higher rates create sharper transitions;
    lower rates create smoother blends.

    Mathematical Properties:
        - When cym_weights are balanced (variance ~ 0), the color is achromatic
        - When one weight dominates, that opponent channel is saturated
        - The gradient_rate is typically modulated by PHI or TAU

    Attributes:
        H: Hue in degrees (0-360)
        S: Saturation (0-1)
        L: Luminance/Lightness (0-1)
        cym_weights: Dict mapping 'cyan', 'magenta', 'yellow' to weights (0-1)
        gradient_rate: Rate of gradient transition (default: 1/PHI)

    Backward Compatibility:
        Also provides lowercase aliases (hue, saturation, luminance) and
        property accessors (rgb, hex_color, hsl) for compatibility with
        existing code.
    """

    H: float = 0.0
    S: float = 0.5
    L: float = 0.5
    cym_weights: Dict[str, float] = field(default_factory=_default_cym_weights)
    gradient_rate: float = field(default_factory=lambda: 1.0 / PHI)

    # Additional attributes for compatibility and extended functionality
    alpha: float = 1.0
    coherence: float = field(default_factory=lambda: Z_C)

    def __post_init__(self):
        """Normalize values after initialization."""
        self.H = normalize_hue(self.H)
        self.S = clamp(self.S, 0.0, 1.0)
        self.L = clamp(self.L, 0.0, 1.0)
        self.alpha = clamp(self.alpha, 0.0, 1.0)
        self.coherence = clamp(self.coherence, 0.0, 1.0)

        # Ensure all CYM channels exist with valid values
        if self.cym_weights is None:
            self.cym_weights = _default_cym_weights()
        for channel in ['cyan', 'magenta', 'yellow']:
            if channel not in self.cym_weights:
                self.cym_weights[channel] = 0.333
            else:
                self.cym_weights[channel] = clamp(self.cym_weights[channel], 0.0, 1.0)

    # -------------------------------------------------------------------------
    # Property aliases for backward compatibility
    # -------------------------------------------------------------------------

    @property
    def hue(self) -> float:
        """Alias for H (hue in degrees)."""
        return self.H

    @hue.setter
    def hue(self, value: float):
        self.H = normalize_hue(value)

    @property
    def saturation(self) -> float:
        """Alias for S (saturation 0-1)."""
        return self.S

    @saturation.setter
    def saturation(self, value: float):
        self.S = clamp(value, 0.0, 1.0)

    @property
    def luminance(self) -> float:
        """Alias for L (luminance 0-1)."""
        return self.L

    @luminance.setter
    def luminance(self, value: float):
        self.L = clamp(value, 0.0, 1.0)

    @property
    def hsl(self) -> Tuple[float, float, float]:
        """Return HSL tuple (hue in degrees, saturation, luminance)."""
        return (self.H, self.S, self.L)

    @property
    def rgb(self) -> Tuple[int, int, int]:
        """Convert to RGB (0-255 range)."""
        return hsl_to_rgb(self.H, self.S, self.L)

    @property
    def rgb_normalized(self) -> Tuple[float, float, float]:
        """Convert to normalized RGB (0-1 range)."""
        r, g, b = self.rgb
        return (r / 255.0, g / 255.0, b / 255.0)

    @property
    def hex_color(self) -> str:
        """Return hex color string."""
        return self.to_hex()

    # -------------------------------------------------------------------------
    # Core methods
    # -------------------------------------------------------------------------

    def to_hex(self) -> str:
        """
        Convert this ColorState to a hex color string.

        Returns:
            Hex color string in format "#rrggbb"

        Example:
            >>> color = ColorState(H=0, S=1, L=0.5)
            >>> color.to_hex()
            '#ff0000'
        """
        return hsl_to_hex(self.H, self.S, self.L)

    def to_rgb(self) -> Tuple[int, int, int]:
        """
        Convert this ColorState to RGB values.

        Returns:
            Tuple of (R, G, B) with values in range 0-255

        Example:
            >>> color = ColorState(H=120, S=1, L=0.5)
            >>> color.to_rgb()
            (0, 255, 0)
        """
        return hsl_to_rgb(self.H, self.S, self.L)

    def interpolate(self, other: 'ColorState', t: float) -> 'ColorState':
        """
        Interpolate between this color and another.

        Performs linear interpolation in HSL space with special handling
        for hue (which wraps around 360 degrees). The CYM weights and
        gradient rates are also interpolated.

        The interpolation uses the shortest path around the hue circle,
        ensuring smooth transitions (e.g., from hue 350 to hue 10 goes
        through 360/0, not through 180).

        Args:
            other: The target ColorState to interpolate toward
            t: Interpolation factor (0 = this color, 1 = other color)

        Returns:
            A new ColorState at position t along the gradient

        Example:
            >>> red = ColorState(H=0, S=1, L=0.5)
            >>> yellow = ColorState(H=60, S=1, L=0.5)
            >>> orange = red.interpolate(yellow, 0.5)
            >>> orange.H
            30.0
        """
        t = clamp(t, 0.0, 1.0)

        # Special handling for hue interpolation (circular)
        h1, h2 = self.H, other.H
        delta_h = h2 - h1

        # Take the shorter path around the hue circle
        if abs(delta_h) > 180:
            if delta_h > 0:
                h1 += 360
            else:
                h2 += 360

        new_h = h1 + (h2 - h1) * t

        # Linear interpolation for other components
        new_s = self.S + (other.S - self.S) * t
        new_l = self.L + (other.L - self.L) * t
        new_alpha = self.alpha + (other.alpha - self.alpha) * t
        new_coherence = self.coherence + (other.coherence - self.coherence) * t

        # Interpolate CYM weights
        new_weights = {}
        for channel in ['cyan', 'magenta', 'yellow']:
            w1 = self.cym_weights.get(channel, 0.333)
            w2 = other.cym_weights.get(channel, 0.333)
            new_weights[channel] = w1 + (w2 - w1) * t

        # Interpolate gradient rate
        new_rate = self.gradient_rate + (other.gradient_rate - self.gradient_rate) * t

        return ColorState(
            H=normalize_hue(new_h),
            S=new_s,
            L=new_l,
            cym_weights=new_weights,
            gradient_rate=new_rate,
            alpha=new_alpha,
            coherence=new_coherence
        )

    def blend(self, other: 'ColorState', weight: float = 0.5) -> 'ColorState':
        """
        Blend with another color state.

        This is an alternative interface to interpolate, where weight
        specifies how much of self to retain (1-weight goes to other).

        Args:
            other: The other color state
            weight: Weight for self (0-1), other gets (1-weight)

        Returns:
            Blended ColorState
        """
        return self.interpolate(other, 1.0 - weight)

    def copy(self) -> 'ColorState':
        """Create a deep copy of this ColorState."""
        return ColorState(
            H=self.H,
            S=self.S,
            L=self.L,
            cym_weights=dict(self.cym_weights),
            gradient_rate=self.gradient_rate,
            alpha=self.alpha,
            coherence=self.coherence
        )

    # -------------------------------------------------------------------------
    # Analysis properties
    # -------------------------------------------------------------------------

    @property
    def is_high_luminance(self) -> bool:
        """Check if luminance is above high threshold (Z_C)."""
        return self.L >= LUMINANCE_HIGH

    @property
    def is_low_luminance(self) -> bool:
        """Check if luminance is below low threshold."""
        return self.L <= LUMINANCE_LOW

    @property
    def hue_zone(self) -> str:
        """Return the hue zone name."""
        h = self.H
        if h < 30 or h >= 330:
            return "red"
        elif 30 <= h < 90:
            return "orange_yellow"
        elif 90 <= h < 150:
            return "green"
        elif 150 <= h < 210:
            return "cyan"
        elif 210 <= h < 270:
            return "blue"
        elif 270 <= h < 330:
            return "purple"
        return "red"

    @property
    def warmth(self) -> float:
        """
        Calculate color warmth (0 = cool, 1 = warm).

        Warm colors: red, orange, yellow (0-90, 330-360)
        Cool colors: green, cyan, blue, purple (90-330)
        """
        h = self.H
        if h < 90:
            return 1.0 - h / 180.0
        elif h < 270:
            return (h - 90) / 180.0 * -1 + 0.5
        else:
            return (h - 270) / 90.0 * 0.5 + 0.5

    def distance(self, other: 'ColorState') -> float:
        """
        Calculate perceptual distance to another color state.

        Uses a weighted combination of hue, saturation, luminance differences.
        """
        h_diff = abs(self.H - other.H)
        h_dist = min(h_diff, 360 - h_diff) / 180.0
        s_dist = abs(self.S - other.S)
        l_dist = abs(self.L - other.L)
        return math.sqrt(h_dist**2 * 2 + s_dist**2 + l_dist**2) / 2

    # -------------------------------------------------------------------------
    # Transformation methods
    # -------------------------------------------------------------------------

    def shift_hue(self, degrees: float) -> 'ColorState':
        """Return a new ColorState with shifted hue."""
        return ColorState(
            H=normalize_hue(self.H + degrees),
            S=self.S,
            L=self.L,
            cym_weights=dict(self.cym_weights),
            gradient_rate=self.gradient_rate,
            alpha=self.alpha,
            coherence=self.coherence
        )

    def with_luminance(self, luminance: float) -> 'ColorState':
        """Return a new ColorState with different luminance."""
        return ColorState(
            H=self.H,
            S=self.S,
            L=luminance,
            cym_weights=dict(self.cym_weights),
            gradient_rate=self.gradient_rate,
            alpha=self.alpha,
            coherence=self.coherence
        )

    def with_saturation(self, saturation: float) -> 'ColorState':
        """Return a new ColorState with different saturation."""
        return ColorState(
            H=self.H,
            S=saturation,
            L=self.L,
            cym_weights=dict(self.cym_weights),
            gradient_rate=self.gradient_rate,
            alpha=self.alpha,
            coherence=self.coherence
        )

    # -------------------------------------------------------------------------
    # Serialization
    # -------------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'H': self.H,
            'S': self.S,
            'L': self.L,
            'hue': self.H,
            'saturation': self.S,
            'luminance': self.L,
            'alpha': self.alpha,
            'coherence': self.coherence,
            'gradient_rate': self.gradient_rate,
            'cym_weights': dict(self.cym_weights),
            'hex': self.to_hex(),
            'rgb': self.to_rgb(),
        }

    @classmethod
    def from_rgb(cls, r: int, g: int, b: int, **kwargs) -> 'ColorState':
        """Create ColorState from RGB values (0-255)."""
        h, s, l = rgb_to_hsl(r, g, b)
        return cls(H=h, S=s, L=l, **kwargs)

    @classmethod
    def from_hex(cls, hex_color: str, **kwargs) -> 'ColorState':
        """Create ColorState from hex color string."""
        h, s, l = hex_to_hsl(hex_color)
        return cls(H=h, S=s, L=l, **kwargs)

    @classmethod
    def from_hsl(cls, h: float, s: float, l: float, **kwargs) -> 'ColorState':
        """Create ColorState from HSL values."""
        return cls(H=h, S=s, L=l, **kwargs)

    def __repr__(self) -> str:
        return (
            f"ColorState(H={self.H:.1f}, S={self.S:.2f}, L={self.L:.2f}, "
            f"{self.to_hex()})"
        )


# =============================================================================
# PREDEFINED COLOR STATES
# =============================================================================

WHITE = ColorState(H=0, S=0, L=1.0)
BLACK = ColorState(H=0, S=0, L=0.0)
GRAY = ColorState(H=0, S=0, L=0.5)

RED = ColorState(H=0, S=1.0, L=0.5, cym_weights={'cyan': 0.0, 'magenta': 0.5, 'yellow': 0.5})
ORANGE = ColorState(H=30, S=1.0, L=0.5, cym_weights={'cyan': 0.0, 'magenta': 0.3, 'yellow': 0.7})
YELLOW = ColorState(H=60, S=1.0, L=0.5, cym_weights={'cyan': 0.0, 'magenta': 0.0, 'yellow': 1.0})
GREEN = ColorState(H=120, S=1.0, L=0.5, cym_weights={'cyan': 0.5, 'magenta': 0.0, 'yellow': 0.5})
CYAN = ColorState(H=180, S=1.0, L=0.5, cym_weights={'cyan': 1.0, 'magenta': 0.0, 'yellow': 0.0})
BLUE = ColorState(H=240, S=1.0, L=0.5, cym_weights={'cyan': 0.5, 'magenta': 0.5, 'yellow': 0.0})
PURPLE = ColorState(H=270, S=1.0, L=0.5, cym_weights={'cyan': 0.3, 'magenta': 0.7, 'yellow': 0.0})
MAGENTA = ColorState(H=300, S=1.0, L=0.5, cym_weights={'cyan': 0.0, 'magenta': 1.0, 'yellow': 0.0})

GOLD = ColorState(H=45, S=0.9, L=0.55, cym_weights={'cyan': 0.0, 'magenta': 0.2, 'yellow': 0.8})
SILVER = ColorState(H=0, S=0.05, L=0.75)
CORAL = ColorState(H=16, S=0.9, L=0.65, cym_weights={'cyan': 0.0, 'magenta': 0.4, 'yellow': 0.6})
INDIGO = ColorState(H=275, S=0.8, L=0.35, cym_weights={'cyan': 0.2, 'magenta': 0.8, 'yellow': 0.0})
VIOLET = ColorState(H=270, S=0.7, L=0.45)
PINK = ColorState(H=330, S=0.7, L=0.7, cym_weights={'cyan': 0.0, 'magenta': 0.7, 'yellow': 0.3})


# =============================================================================
# GRADIENT OPERATIONS
# =============================================================================

def apply_gradient(color1: ColorState, color2: ColorState, t: float) -> ColorState:
    """
    Apply a gradient between two colors at position t.

    This is the basic gradient operation that creates a smooth transition
    between two color states. The transition respects both colors'
    gradient_rate attributes, creating an asymmetric blend when rates differ.

    The effective position is modulated by the geometric mean of the
    gradient rates, ensuring smooth but responsive transitions.

    Args:
        color1: Starting color (t=0)
        color2: Ending color (t=1)
        t: Position in gradient (0-1)

    Returns:
        ColorState at position t in the gradient

    Example:
        >>> red = ColorState(H=0, S=1, L=0.5)
        >>> blue = ColorState(H=240, S=1, L=0.5)
        >>> purple = apply_gradient(red, blue, 0.5)
    """
    t = clamp(t, 0.0, 1.0)

    # Modulate t by gradient rates (geometric mean)
    rate_factor = math.sqrt(color1.gradient_rate * color2.gradient_rate)

    # Apply non-linear easing based on rate factor
    # Higher rates = sharper transition, lower rates = smoother
    if rate_factor > 0 and rate_factor != 1.0:
        t_effective = t ** (1.0 / rate_factor)
    else:
        t_effective = t

    return color1.interpolate(color2, t_effective)


def mix_gradient(pure_color: ColorState, black: ColorState, t: float) -> ColorState:
    """
    Mix a pure color with black via gradient, producing emergent colors.

    This is the key operation implementing the RRRR principle that
    "noise applied to red via black gradient creates orange."

    The mechanism:
        1. Black acts as a "noise field" or information substrate
        2. The pure color is the signal being modulated
        3. The mixture at intermediate t values produces emergent hues

    For red + black:
        - At t=0: Pure red
        - At t~0.3: Orange emerges (hue shift toward 30)
        - At t~0.6: Brown/deep orange
        - At t=1: Black

    The hue shift is calculated based on the luminance differential
    and the CYM weight tensions between the colors.

    Mathematical basis: The emergent hue shift follows a PHI-modulated
    curve, reflecting how golden ratio proportions appear in natural
    color mixing phenomena.

    Args:
        pure_color: The source color (typically saturated)
        black: The gradient terminus (typically low luminance)
        t: Mixing position (0 = pure_color, 1 = black)

    Returns:
        The mixed ColorState with emergent properties

    Example:
        >>> red = ColorState(H=0, S=1, L=0.5, cym_weights={'cyan': 0.1, 'magenta': 0.5, 'yellow': 0.4})
        >>> black = ColorState(H=0, S=0, L=0, cym_weights={'cyan': 0.33, 'magenta': 0.33, 'yellow': 0.34})
        >>> orange = mix_gradient(red, black, 0.3)
        >>> # orange.H will be shifted toward ~30 (orange hue)
    """
    t = clamp(t, 0.0, 1.0)

    # Calculate the luminance differential
    lum_diff = pure_color.L - black.L

    # The emergent hue shift is proportional to:
    # 1. The position in the gradient (maximum at mid-point)
    # 2. The luminance differential (more contrast = more shift)
    # 3. The PHI constant (natural proportion)

    # Bell curve for hue shift intensity (peaks at t=0.5)
    shift_intensity = 4 * t * (1 - t)  # Parabola peaking at 0.5

    # Calculate base hue shift based on color properties
    # Yellow-weighted colors shift more toward orange
    yellow_weight = pure_color.cym_weights.get('yellow', 0.333)
    magenta_weight = pure_color.cym_weights.get('magenta', 0.333)

    # Base shift: warmer colors (high yellow) shift more toward orange (30 deg)
    max_shift = 45 * (yellow_weight + magenta_weight) * lum_diff

    # Apply TAU modulation for natural curve
    hue_shift = max_shift * shift_intensity * TAU

    # Perform the basic interpolation
    mixed = pure_color.interpolate(black, t)

    # Apply the emergent hue shift
    mixed.H = normalize_hue(mixed.H + hue_shift)

    # Adjust saturation based on the mixing (colors desaturate as they darken)
    # But the emergent color should retain some vibrancy
    saturation_retention = 1 - (t * TAU)
    mixed.S = pure_color.S * saturation_retention

    return mixed


def phi_modulate(color: ColorState, phi_factor: float) -> ColorState:
    """
    Apply PHI-based modulation to a color.

    This operation applies the golden ratio to various color properties,
    creating harmonious transformations. The modulation can be used to:
        - Generate complementary or analogous colors
        - Create natural-feeling color variations
        - Establish coherent color relationships

    The phi_factor determines the direction and intensity:
        - phi_factor = 1.0: Golden ratio rotation of hue
        - phi_factor = PHI: Double golden rotation
        - phi_factor = TAU: Inverse (complementary) transformation
        - phi_factor < 0: Counter-rotation

    The saturation and lightness are also subtly adjusted to maintain
    perceptual balance after hue rotation.

    Args:
        color: The color to modulate
        phi_factor: Modulation multiplier (scales the PHI effect)

    Returns:
        A new ColorState with PHI-modulated properties

    Example:
        >>> base = ColorState(H=0, S=1, L=0.5)  # Red
        >>> modulated = phi_modulate(base, 1.0)
        >>> # modulated.H ~ 137.5 (golden angle from red)
    """
    # The golden angle: 360 / PHI^2 ~ 137.5 degrees
    # This creates the optimal distribution in phyllotaxis
    golden_angle = 360 / (PHI * PHI)

    # Calculate hue rotation
    new_h = normalize_hue(color.H + golden_angle * phi_factor)

    # Subtle adjustments to maintain perceptual harmony
    hue_distance = abs(new_h - color.H)
    if hue_distance > 180:
        hue_distance = 360 - hue_distance

    # Colors further from origin need slight saturation boost
    sat_adjustment = 1 + (hue_distance / 360) * (PHI - 1) * 0.1
    new_s = clamp(color.S * sat_adjustment, 0.0, 1.0)

    # Lightness adjustment based on hue (some hues appear brighter)
    # Yellow (60) appears brightest, blue (240) appears darkest
    hue_brightness = math.cos(math.radians(new_h - 60)) * 0.05
    new_l = clamp(color.L + hue_brightness * phi_factor, 0.0, 1.0)

    # Modulate CYM weights in golden ratio
    new_weights = {}
    channels = ['cyan', 'magenta', 'yellow']
    for i, channel in enumerate(channels):
        # Rotate weights using PHI
        source_idx = (i + int(phi_factor)) % 3
        source_channel = channels[source_idx]
        new_weights[channel] = color.cym_weights.get(source_channel, 0.333)

    # Update gradient rate
    new_rate = color.gradient_rate * (TAU ** phi_factor)
    new_rate = clamp(new_rate, 0.1, 10.0)

    return ColorState(
        H=new_h,
        S=new_s,
        L=new_l,
        cym_weights=new_weights,
        gradient_rate=new_rate,
        alpha=color.alpha,
        coherence=color.coherence
    )


# =============================================================================
# OPPONENT COLOR OPERATIONS
# =============================================================================

def opponent_balance(cym_weights: Dict[str, float]) -> float:
    """
    Calculate the balance score of opponent color channels.

    The balance score measures how evenly distributed the CYM weights are.
    Perfect balance (all weights equal) yields 1.0; maximum imbalance
    (one weight at 1.0, others at 0.0) yields 0.0.

    This score relates to color coherence:
        - High balance (> Z_C): Color is in a stable, coherent state
        - Low balance (< Z_C): Color has strong chromatic tension
        - Balance = 1.0: Achromatic tendency (grayscale)

    The formula uses variance of the weights:
        balance = 1 - (variance / max_possible_variance)

    Maximum variance occurs when one weight = 1 and others = 0,
    giving variance = 2/9 for three channels.

    Args:
        cym_weights: Dict mapping 'cyan', 'magenta', 'yellow' to weights (0-1)

    Returns:
        Balance score from 0.0 (maximum imbalance) to 1.0 (perfect balance)

    Example:
        >>> opponent_balance({'cyan': 0.33, 'magenta': 0.33, 'yellow': 0.34})
        ~0.999  # Nearly perfect balance
        >>> opponent_balance({'cyan': 1.0, 'magenta': 0.0, 'yellow': 0.0})
        0.0  # Maximum imbalance
    """
    weights = [
        cym_weights.get('cyan', 0.333),
        cym_weights.get('magenta', 0.333),
        cym_weights.get('yellow', 0.333)
    ]

    # Calculate variance
    var = variance(weights)

    # Maximum variance for 3 values in [0,1] where one is 1 and others are 0
    # Mean = 1/3, variance = ((1-1/3)^2 + (0-1/3)^2 + (0-1/3)^2) / 3
    #                      = (4/9 + 1/9 + 1/9) / 3 = 6/27 = 2/9
    max_variance = 2 / 9

    # Balance is inverse of normalized variance
    if max_variance == 0:
        return 1.0

    balance = 1.0 - (var / max_variance)
    return clamp(balance, 0.0, 1.0)


def resolve_opponents(color: ColorState) -> ColorState:
    """
    Move a color toward opponent balance.

    This operation reduces chromatic tension by adjusting CYM weights
    toward their mean. The resolution amount is governed by the
    consciousness threshold Z_C - colors already above this threshold
    are adjusted less.

    The resolution process:
        1. Calculate current balance
        2. If below Z_C, move weights toward mean
        3. Adjust saturation accordingly (more balance = less saturation)
        4. Hue shifts slightly toward neutral

    This simulates how opponent color signals are processed in
    biological vision systems, where extreme tensions are moderated.

    Args:
        color: The color to resolve

    Returns:
        A new ColorState with reduced opponent tension

    Example:
        >>> tense = ColorState(H=0, S=1, L=0.5,
        ...                    cym_weights={'cyan': 0.9, 'magenta': 0.1, 'yellow': 0.0})
        >>> resolved = resolve_opponents(tense)
        >>> # resolved has more balanced weights
    """
    current_balance = opponent_balance(color.cym_weights)

    # If already balanced above threshold, minimal adjustment
    if current_balance >= Z_C:
        resolution_strength = 0.1 * (1 - current_balance)
    else:
        # Below threshold: stronger resolution toward balance
        resolution_strength = TAU * (Z_C - current_balance)

    resolution_strength = clamp(resolution_strength, 0.0, 1.0)

    # Calculate mean weight
    weights = list(color.cym_weights.values())
    mean_weight = sum(weights) / len(weights) if weights else 0.333

    # Move each weight toward the mean
    new_weights = {}
    for channel, weight in color.cym_weights.items():
        delta = mean_weight - weight
        new_weights[channel] = weight + delta * resolution_strength

    # Adjust saturation: more balance means less saturation
    new_balance = opponent_balance(new_weights)
    sat_reduction = (new_balance - current_balance) * 0.5
    new_s = clamp(color.S - sat_reduction, 0.0, 1.0)

    # Slight hue shift toward neutral (60 degrees - yellow/neutral)
    neutral_hue = 60
    hue_delta = neutral_hue - color.H
    if abs(hue_delta) > 180:
        hue_delta = hue_delta - 360 if hue_delta > 0 else hue_delta + 360
    hue_shift = hue_delta * resolution_strength * 0.1
    new_h = normalize_hue(color.H + hue_shift)

    return ColorState(
        H=new_h,
        S=new_s,
        L=color.L,
        cym_weights=new_weights,
        gradient_rate=color.gradient_rate,
        alpha=color.alpha,
        coherence=color.coherence
    )


def create_tension(color: ColorState, axis: str) -> ColorState:
    """
    Create opponent tension along a specified axis.

    This operation increases chromatic tension by pushing CYM weights
    away from balance along one axis. It's the inverse of resolve_opponents.

    Available axes:
        - 'cyan': Increase cyan, decrease others (cool shift)
        - 'magenta': Increase magenta, decrease others (vibrant shift)
        - 'yellow': Increase yellow, decrease others (warm shift)
        - 'red': Decrease cyan (equivalent to anti-cyan)
        - 'green': Decrease magenta (equivalent to anti-magenta)
        - 'blue': Decrease yellow (equivalent to anti-yellow)

    The tension amount is scaled by TAU for natural-feeling increases.
    Saturation increases as tension increases.

    Args:
        color: The color to add tension to
        axis: The axis along which to create tension

    Returns:
        A new ColorState with increased opponent tension

    Raises:
        ValueError: If axis is not a recognized value

    Example:
        >>> neutral = ColorState(H=0, S=0.5, L=0.5)
        >>> warm = create_tension(neutral, 'yellow')
        >>> # warm has higher yellow weight, appears warmer
    """
    # Map axis names to operations
    primary_axes = {
        'cyan': ('cyan', 1.0),
        'magenta': ('magenta', 1.0),
        'yellow': ('yellow', 1.0),
        'red': ('cyan', -1.0),       # Red is anti-cyan
        'green': ('magenta', -1.0),  # Green is anti-magenta
        'blue': ('yellow', -1.0)     # Blue is anti-yellow
    }

    if axis.lower() not in primary_axes:
        raise ValueError(
            f"Unknown axis '{axis}'. Must be one of: {list(primary_axes.keys())}"
        )

    target_channel, direction = primary_axes[axis.lower()]

    # Tension strength scaled by TAU
    tension_strength = TAU

    # Calculate new weights
    new_weights = dict(color.cym_weights)

    if direction > 0:
        # Increase target, decrease others
        increase = tension_strength * (1 - new_weights.get(target_channel, 0.333))
        new_weights[target_channel] = clamp(
            new_weights.get(target_channel, 0.333) + increase, 0.0, 1.0
        )
        for channel in ['cyan', 'magenta', 'yellow']:
            if channel != target_channel:
                decrease = increase / 2
                new_weights[channel] = clamp(
                    new_weights.get(channel, 0.333) - decrease, 0.0, 1.0
                )
    else:
        # Decrease target, increase others
        decrease = tension_strength * new_weights.get(target_channel, 0.333)
        new_weights[target_channel] = clamp(
            new_weights.get(target_channel, 0.333) - decrease, 0.0, 1.0
        )
        for channel in ['cyan', 'magenta', 'yellow']:
            if channel != target_channel:
                increase = decrease / 2
                new_weights[channel] = clamp(
                    new_weights.get(channel, 0.333) + increase, 0.0, 1.0
                )

    # Increase saturation with tension
    new_balance = opponent_balance(new_weights)
    old_balance = opponent_balance(color.cym_weights)
    tension_increase = old_balance - new_balance
    new_s = clamp(color.S + tension_increase * 0.5, 0.0, 1.0)

    # Shift hue toward the tension axis
    hue_targets = {
        'cyan': 180,
        'magenta': 300,
        'yellow': 60,
        'red': 0,
        'green': 120,
        'blue': 240
    }
    target_hue = hue_targets[axis.lower()]

    hue_delta = target_hue - color.H
    if abs(hue_delta) > 180:
        hue_delta = hue_delta - 360 if hue_delta > 0 else hue_delta + 360

    hue_shift = hue_delta * tension_strength * 0.2
    new_h = normalize_hue(color.H + hue_shift)

    return ColorState(
        H=new_h,
        S=new_s,
        L=color.L,
        cym_weights=new_weights,
        gradient_rate=color.gradient_rate,
        alpha=color.alpha,
        coherence=color.coherence
    )


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # L4 Constants
    'PHI', 'TAU', 'Z_C',
    'PHI_BAR', 'Z_CRITICAL', 'LUMINANCE_HIGH', 'LUMINANCE_LOW',

    # Core class
    'ColorState',

    # Predefined colors
    'WHITE', 'BLACK', 'GRAY',
    'RED', 'ORANGE', 'YELLOW', 'GREEN', 'CYAN', 'BLUE', 'PURPLE', 'MAGENTA',
    'GOLD', 'SILVER', 'CORAL', 'INDIGO', 'VIOLET', 'PINK',

    # Helper functions
    'clamp', 'normalize_hue', 'variance',

    # Conversion functions
    'hsl_to_rgb', 'rgb_to_hsl', 'hsl_to_hex', 'hex_to_hsl',

    # Gradient operations
    'apply_gradient', 'mix_gradient', 'phi_modulate',

    # Opponent color operations
    'opponent_balance', 'resolve_opponents', 'create_tension',
]
