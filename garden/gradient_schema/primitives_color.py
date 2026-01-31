"""
Primitives Color - Mapping Six Computational Primitives to Color Operations

This module implements the theoretical connection between the Six Computational
Primitives from Jordan Normal Form and visual color operations in the RRRR
gradient schema system.

=============================================================================
JORDAN NORMAL FORM AND THE SIX PRIMITIVES
=============================================================================

The Six Primitives emerge from the complete classification of linear operators
via Jordan Normal Form. Any 2x2 matrix can be decomposed into Jordan blocks,
and the eigenvalue structure determines the primitive type:

DIAGONALIZABLE MATRICES (k=1, single Jordan blocks):
-----------------------------------------------------

1. FIX (|lambda| < 1):
   - Eigenvalue magnitude less than 1
   - Attractive fixed point dynamics
   - Repeated application converges to fixed point
   - Color Effect: Convergence toward neutral gray (desaturation,
     lightness moves to 0.5)

2. REPEL (|lambda| > 1):
   - Eigenvalue magnitude greater than 1
   - Repulsive fixed point dynamics
   - Divergence from equilibrium
   - Color Effect: Explosion from neutral (saturation increase,
     contrast amplification)

3. INV (|lambda| = 1, complex):
   - Unit circle eigenvalues with imaginary component
   - Rotation/oscillation dynamics
   - Periodic orbits in phase space
   - Color Effect: Hue rotation (cycling through the color wheel)

4. OSC (mixed eigenvalues):
   - Mixed eigenvalue magnitudes
   - Combination of convergent and divergent modes
   - Quasiperiodic or transient dynamics
   - Color Effect: Pulsing/breathing of color properties

NON-DIAGONALIZABLE MATRICES (k>1, Jordan blocks with ones above diagonal):
---------------------------------------------------------------------------

5. HALT (lambda=1 repeated, k>1):
   - Jordan block with eigenvalue 1
   - Parabolic fixed point (critical boundary)
   - Linear growth in components
   - Color Effect: Edge sharpening, contrast bands (the "critical
     point" between convergent and divergent)

6. MIX (lambda=0 repeated, k>1):
   - Nilpotent matrix (M^k = 0)
   - Information destruction
   - Irreversible transformation
   - Color Effect: Noise injection, information loss (the
     "measurement" operation in quantum terms)

=============================================================================
SIGNATURE SPACE
=============================================================================

The signature (sigma_FIX, sigma_REPEL, sigma_INV, sigma_OSC, sigma_HALT, sigma_MIX)
forms a 6-dimensional space that characterizes any color state's relationship
to the six primitives. The signature is normalized so components sum to 1.

The dominant primitive indicates the "character" of a color state:
- FIX-dominant: Muted, calm colors (low saturation, balanced luminance)
- REPEL-dominant: Vivid, intense colors (high saturation)
- INV-dominant: Active, dynamic hues (high hue activity)
- OSC-dominant: Transitional states (mixed characteristics)
- HALT-dominant: High contrast, edge states (extreme luminance)
- MIX-dominant: Chaotic, noisy states (unbalanced opponent weights)

=============================================================================
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Tuple, Optional, List

from .color_ops import (
    ColorState,
    PHI,
    TAU,
    Z_C,
    clamp,
    normalize_hue,
    variance,
)

# =============================================================================
# Constants
# =============================================================================

# Time constant for FIX decay (controls convergence rate)
TAU_FIX = 0.5

# Golden angle for hue rotation (relates to INV)
GOLDEN_ANGLE = 360.0 / (PHI * PHI)  # ~137.5 degrees

# Noise amplitude for MIX operation
DEFAULT_NOISE_LEVEL = 0.1

# Number of contrast bands for HALT operation
HALT_BANDS = 5


# =============================================================================
# PrimitiveSignature Dataclass
# =============================================================================

@dataclass
class PrimitiveSignature:
    """
    Represents the signature of a color state in primitive space.

    The signature captures how much a color state exhibits characteristics
    of each of the six computational primitives. This forms a probability
    distribution over primitive types.

    Attributes:
        fix: sigma_FIX - Convergence toward neutral (|lambda| < 1)
        repel: sigma_REPEL - Divergence from neutral (|lambda| > 1)
        inv: sigma_INV - Hue rotation activity (|lambda| = 1, complex)
        osc: sigma_OSC - Gradient oscillation (mixed eigenvalues)
        halt: sigma_HALT - Edge/contrast sharpening (lambda=1, Jordan k>1)
        mix: sigma_MIX - Information destruction (lambda=0, Jordan k>1)

    Mathematical Interpretation:
        The signature can be viewed as a projection of the color state onto
        the six primitive eigenvectors in the space of color dynamics. High
        values indicate the color state will respond strongly to that
        primitive's operation.

    Jordan Normal Form Connection:
        - fix + repel: Diagonalizable, real eigenvalues
        - inv: Diagonalizable, complex unit eigenvalues
        - osc: Diagonalizable, mixed eigenvalue magnitudes
        - halt + mix: Non-diagonalizable (Jordan blocks with k > 1)
    """
    fix: float    # sigma_FIX: Convergence to neutral
    repel: float  # sigma_REPEL: Divergence from neutral
    inv: float    # sigma_INV: Hue rotation
    osc: float    # sigma_OSC: Oscillation/pulsing (optional, can combine with mix)
    halt: float   # sigma_HALT: Edge sharpening (optional)
    mix: float    # sigma_MIX: Noise/information destruction

    def __post_init__(self):
        """Ensure non-negative values."""
        self.fix = max(0.0, self.fix)
        self.repel = max(0.0, self.repel)
        self.inv = max(0.0, self.inv)
        self.osc = max(0.0, self.osc)
        self.halt = max(0.0, self.halt)
        self.mix = max(0.0, self.mix)

    def normalize(self) -> 'PrimitiveSignature':
        """
        Normalize the signature so all components sum to 1.

        Returns:
            New PrimitiveSignature with normalized values forming a
            probability distribution over primitive types.
        """
        total = self.fix + self.repel + self.inv + self.osc + self.halt + self.mix
        if total == 0:
            # Default to uniform distribution if all zero
            return PrimitiveSignature(
                fix=1/6, repel=1/6, inv=1/6, osc=1/6, halt=1/6, mix=1/6
            )
        return PrimitiveSignature(
            fix=self.fix / total,
            repel=self.repel / total,
            inv=self.inv / total,
            osc=self.osc / total,
            halt=self.halt / total,
            mix=self.mix / total
        )

    def dominant(self) -> str:
        """
        Return the name of the dominant primitive.

        The dominant primitive is the one with the highest signature value,
        indicating which dynamical behavior most characterizes the color state.

        Returns:
            One of: 'FIX', 'REPEL', 'INV', 'OSC', 'HALT', 'MIX'
        """
        values = [
            ('FIX', self.fix),
            ('REPEL', self.repel),
            ('INV', self.inv),
            ('OSC', self.osc),
            ('HALT', self.halt),
            ('MIX', self.mix)
        ]
        return max(values, key=lambda x: x[1])[0]

    def to_vector(self) -> Tuple[float, float, float, float, float, float]:
        """Return signature as a 6-tuple vector."""
        return (self.fix, self.repel, self.inv, self.osc, self.halt, self.mix)

    def diagonalizable_content(self) -> float:
        """
        Fraction of signature in diagonalizable primitives.

        In Jordan Normal Form, diagonalizable matrices have k=1 (no
        superdiagonal 1s). This includes FIX, REPEL, INV, and OSC.

        Returns:
            Value in [0, 1] indicating quantum-like (reversible) content.
        """
        norm = self.normalize()
        return norm.fix + norm.repel + norm.inv + norm.osc

    def jordan_content(self) -> float:
        """
        Fraction of signature in non-diagonalizable (Jordan k>1) primitives.

        Non-diagonalizable matrices have Jordan blocks with superdiagonal 1s.
        This includes HALT and MIX, representing classical/irreversible behavior.

        Returns:
            Value in [0, 1] indicating classical-like (irreversible) content.
        """
        norm = self.normalize()
        return norm.halt + norm.mix

    def __repr__(self) -> str:
        return (f"PrimitiveSignature(fix={self.fix:.3f}, repel={self.repel:.3f}, "
                f"inv={self.inv:.3f}, osc={self.osc:.3f}, halt={self.halt:.3f}, "
                f"mix={self.mix:.3f}) [{self.dominant()}]")


# =============================================================================
# Signature Computation
# =============================================================================

def compute_color_signature(
    H: float,
    S: float,
    L: float,
    cym_weights: Optional[Tuple[float, float, float]] = None
) -> PrimitiveSignature:
    """
    Compute the primitive signature for a color state.

    This function maps a color's HSL values and opponent channel weights
    to a signature in primitive space, indicating how the color relates
    to each of the six computational primitives.

    Args:
        H: Hue in degrees [0, 360)
        S: Saturation [0, 1]
        L: Lightness [0, 1]
        cym_weights: Optional tuple of (cyan, yellow, magenta) opponent
                     channel weights, each in [0, 1]. If None, defaults
                     to balanced weights (0.5, 0.5, 0.5).

    Returns:
        PrimitiveSignature with computed values for each primitive.

    Signature Formulas:
        sigma_FIX = exp(-gray_dist / tau)
            High when close to neutral gray (low saturation, mid luminance).
            The exponential decay models the attractive basin of the FIX
            fixed point.

        sigma_REPEL = S * gray_dist
            High when far from neutral with high saturation. Product of
            distance from neutral and saturation captures the "explosive"
            character of vivid colors.

        sigma_INV = hue_activity * (1 - |L - 0.5| * 2)
            High when hue is perceptually active (high saturation, mid
            luminance). The hue rotation is most visible when not washed
            out by extreme lightness.

        sigma_OSC = S * (1 - S) * 4 * lightness_activity
            Peaks at intermediate saturation and luminance, representing
            the transitional zone where oscillatory dynamics dominate.

        sigma_HALT = |L - 0.5| * 2 * (1 - S)
            High at extreme luminance (near black or white) with low
            saturation. These are the "edge" states where contrast bands
            become visible.

        sigma_MIX = weight_variance
            High when opponent channel weights are unbalanced. Unbalanced
            weights indicate a state susceptible to noise injection and
            information loss.

    Jordan Normal Form Interpretation:
        - High (FIX + REPEL): Color near real eigenvalue dynamics
        - High INV: Color exhibits rotation (complex eigenvalue)
        - High OSC: Mixed/transient dynamics
        - High (HALT + MIX): Non-diagonalizable, irreversible dynamics
    """
    # Default to balanced opponent weights
    if cym_weights is None:
        cym_weights = (0.5, 0.5, 0.5)

    # Compute gray distance (distance from neutral gray at S=0, L=0.5)
    lightness_deviation = abs(L - 0.5)
    gray_dist = math.sqrt(S ** 2 + lightness_deviation ** 2)

    # Hue activity: how "visible" is the hue?
    # Hue is most active at high saturation and mid-luminance
    lightness_factor = 1.0 - lightness_deviation * 2
    hue_activity = S * max(0.0, lightness_factor)

    # Opponent weight variance
    weight_var = variance(list(cym_weights))

    # Compute signature components

    # FIX: Convergence to neutral (high when close to gray)
    # Uses exponential decay from gray point
    sigma_fix = math.exp(-gray_dist / TAU_FIX)

    # REPEL: Divergence from neutral (high when saturated and far from gray)
    sigma_repel = S * gray_dist

    # INV: Hue rotation (high when hue is perceptually active)
    sigma_inv = hue_activity * max(0.0, 1.0 - lightness_deviation * 2)

    # OSC: Oscillation (peaks at intermediate saturation and luminance)
    # This captures the "breathing" quality of transitional colors
    lightness_activity = 1.0 - abs(L - 0.5) * 2
    sigma_osc = S * (1.0 - S) * 4 * max(0.0, lightness_activity)

    # HALT: Edge sharpening (high at extreme luminance with low saturation)
    # These are the critical "edge" states at luminance boundaries
    sigma_halt = lightness_deviation * 2 * (1.0 - S)

    # MIX: Information destruction (high when weights are unbalanced)
    # Unbalanced opponent channels indicate noise susceptibility
    # Scale to reasonable range (max variance for 3 values is ~0.22)
    sigma_mix = weight_var * 4.5

    return PrimitiveSignature(
        fix=sigma_fix,
        repel=sigma_repel,
        inv=sigma_inv,
        osc=sigma_osc,
        halt=sigma_halt,
        mix=sigma_mix
    )


def compute_signature_from_color(color: ColorState) -> PrimitiveSignature:
    """
    Convenience function to compute signature directly from a ColorState.

    Args:
        color: ColorState object

    Returns:
        PrimitiveSignature for the color
    """
    # ColorState uses 'luminance' instead of 'lightness'
    return compute_color_signature(
        H=color.hue,
        S=color.saturation,
        L=color.luminance,
        cym_weights=None  # ColorState doesn't have CYM weights
    )


def hsl_to_signature(h: float, s: float, l: float) -> PrimitiveSignature:
    """
    Compute primitive signature from HSL color values.

    This is a convenience function that wraps compute_color_signature
    for direct HSL input. Used for color-to-primitive mapping.

    Args:
        h: Hue in degrees [0, 360)
        s: Saturation [0, 1]
        l: Lightness [0, 1]

    Returns:
        PrimitiveSignature for the HSL color
    """
    return compute_color_signature(H=h, S=s, L=l)


# =============================================================================
# Color Primitive Operations
# =============================================================================

def apply_fix(color: ColorState, strength: float = 0.5) -> ColorState:
    """
    Apply FIX primitive: Converge toward neutral gray.

    The FIX primitive represents attractive fixed point dynamics where
    |lambda| < 1. Repeated application drives the color toward the
    neutral gray fixed point at (H=any, S=0, L=0.5).

    Mathematical Model:
        x_{n+1} = (1 - strength) * x_n + strength * x_fixed
        where x_fixed = neutral gray

    Visual Effect:
        - Saturation decreases (fading toward gray)
        - Luminance moves toward 0.5 (mid-gray)
        - Hue preserved but becomes irrelevant as S -> 0

    Jordan Normal Form:
        Corresponds to matrices with all |eigenvalues| < 1
        (contractive dynamics).

    Args:
        color: Input ColorState
        strength: Convergence strength in [0, 1]. Higher values give
                  faster convergence to gray.

    Returns:
        New ColorState moved toward neutral gray.
    """
    strength = clamp(strength, 0.0, 1.0)

    # Converge saturation toward 0
    new_sat = color.saturation * (1.0 - strength)

    # Converge luminance toward 0.5
    new_lum = color.luminance + (0.5 - color.luminance) * strength

    return ColorState(
        H=color.hue,
        S=new_sat,
        L=new_lum,
        alpha=color.alpha,
        coherence=color.coherence * (1.0 - strength * 0.1),  # Slight coherence loss
        gradient_rate=color.gradient_rate
    )


def apply_repel(color: ColorState, strength: float = 0.5) -> ColorState:
    """
    Apply REPEL primitive: Diverge from neutral, increase saturation.

    The REPEL primitive represents repulsive fixed point dynamics where
    |lambda| > 1. Application pushes the color away from neutral gray,
    creating "saturation explosion."

    Mathematical Model:
        Inverse of FIX dynamics - exponential growth away from fixed point
        x_{n+1} = x_n + strength * (x_n - x_fixed)

    Visual Effect:
        - Saturation increases (colors become more vivid)
        - Luminance pushed away from 0.5 (darker or lighter)
        - Hue becomes more pronounced

    Jordan Normal Form:
        Corresponds to matrices with all |eigenvalues| > 1
        (expansive dynamics).

    Args:
        color: Input ColorState
        strength: Divergence strength in [0, 1]. Higher values give
                  more dramatic saturation increase.

    Returns:
        New ColorState with increased saturation/contrast.
    """
    strength = clamp(strength, 0.0, 1.0)

    # Push saturation toward 1
    new_sat = color.saturation + (1.0 - color.saturation) * strength

    # Push luminance away from 0.5 (amplify contrast)
    lum_deviation = color.luminance - 0.5
    new_lum = clamp(color.luminance + lum_deviation * strength, 0.0, 1.0)

    return ColorState(
        H=color.hue,
        S=new_sat,
        L=new_lum,
        alpha=color.alpha,
        coherence=color.coherence,
        gradient_rate=color.gradient_rate * (1.0 + strength)  # Increase rate
    )


def apply_inv(color: ColorState, angle: float = GOLDEN_ANGLE) -> ColorState:
    """
    Apply INV primitive: Rotate hue by specified angle.

    The INV primitive represents rotational dynamics where |lambda| = 1
    with complex eigenvalues. Application rotates the color around the
    hue circle, creating color cycling effects.

    Mathematical Model:
        Complex rotation: z_{n+1} = e^{i*theta} * z_n
        In color space: H_{n+1} = (H_n + angle) mod 360

    Visual Effect:
        - Hue shifts around the color wheel
        - Saturation and luminance preserved
        - Periodic orbit through the color spectrum

    Jordan Normal Form:
        Corresponds to 2x2 rotation matrices with eigenvalues
        e^{+/- i*theta} on the unit circle.

    Args:
        color: Input ColorState
        angle: Rotation angle in degrees. Default is the golden angle
               (~137.5 degrees), which produces aesthetically pleasing
               distributions around the color wheel.

    Returns:
        New ColorState with rotated hue.
    """
    new_hue = normalize_hue(color.hue + angle)

    return ColorState(
        H=new_hue,
        S=color.saturation,
        L=color.luminance,
        alpha=color.alpha,
        coherence=color.coherence,
        gradient_rate=color.gradient_rate
    )


def apply_osc(
    color: ColorState,
    t: float,
    frequency: float = 1.0,
    amplitude: float = 0.2
) -> ColorState:
    """
    Apply OSC primitive: Oscillate color properties.

    The OSC primitive represents mixed eigenvalue dynamics where some
    components grow while others decay, creating oscillatory or
    quasiperiodic behavior.

    Mathematical Model:
        Damped oscillator: x(t) = A * sin(2*pi*f*t) * e^{-gamma*t}
        Applied to saturation and luminance with phase offsets.

    Visual Effect:
        - Pulsing/breathing of saturation
        - Gentle luminance oscillation
        - Creates organic, "living" color feel

    Jordan Normal Form:
        Corresponds to matrices with mixed eigenvalue magnitudes
        (some |lambda| < 1, some |lambda| > 1).

    Args:
        color: Input ColorState
        t: Time parameter for the oscillation phase
        frequency: Oscillation frequency (cycles per unit time)
        amplitude: Oscillation amplitude (fraction of range)

    Returns:
        New ColorState with oscillated properties.
    """
    amplitude = clamp(amplitude, 0.0, 0.5)

    # Phase-shifted oscillations
    phase = TAU * frequency * t
    sat_osc = amplitude * math.sin(phase)
    lum_osc = amplitude * 0.5 * math.sin(phase + TAU / 4)  # 90 degree offset

    new_sat = clamp(color.saturation + sat_osc, 0.0, 1.0)
    new_lum = clamp(color.luminance + lum_osc, 0.0, 1.0)

    # Gentle hue wobble
    hue_osc = amplitude * 30 * math.sin(phase * PHI)  # Irrational frequency
    new_hue = normalize_hue(color.hue + hue_osc)

    return ColorState(
        H=new_hue,
        S=new_sat,
        L=new_lum,
        alpha=color.alpha,
        coherence=clamp(color.coherence + 0.1 * math.sin(phase), 0.0, 1.0),
        gradient_rate=color.gradient_rate
    )


def apply_halt(color: ColorState, bands: int = HALT_BANDS) -> ColorState:
    """
    Apply HALT primitive: Sharpen contrast, create bands.

    The HALT primitive represents the parabolic fixed point where
    lambda = 1 with Jordan block k > 1. This is the critical boundary
    between convergent and divergent dynamics.

    Mathematical Model:
        Jordan block [[1, 1], [0, 1]] gives linear (not exponential) growth.
        Applied as quantization of luminance into discrete bands.

    Visual Effect:
        - Luminance quantized into discrete bands
        - Creates hard contrast edges
        - Posterization effect

    Jordan Normal Form:
        Corresponds to defective matrices with repeated eigenvalue 1
        and non-trivial Jordan block (the "critical case").

    Args:
        color: Input ColorState
        bands: Number of luminance bands for quantization

    Returns:
        New ColorState with quantized luminance.
    """
    bands = max(2, bands)

    # Quantize luminance into bands
    band_size = 1.0 / bands
    band_index = int(color.luminance / band_size)
    band_index = min(band_index, bands - 1)  # Handle edge case L=1.0

    # Place luminance at band center
    new_lum = (band_index + 0.5) * band_size

    # Boost saturation slightly at band edges
    distance_to_edge = abs((color.luminance / band_size) - band_index - 0.5)
    sat_boost = distance_to_edge * 0.2
    new_sat = clamp(color.saturation + sat_boost, 0.0, 1.0)

    return ColorState(
        H=color.hue,
        S=new_sat,
        L=new_lum,
        alpha=color.alpha,
        coherence=max(0.0, color.coherence - 0.1),  # Slight coherence loss
        gradient_rate=0.0  # Halt stops gradient
    )


def apply_mix(
    color: ColorState,
    noise_level: float = DEFAULT_NOISE_LEVEL,
    seed: Optional[int] = None
) -> ColorState:
    """
    Apply MIX primitive: Inject noise, destroy information.

    The MIX primitive represents nilpotent dynamics where lambda = 0
    with Jordan block k > 1. This corresponds to irreversible information
    destruction - the "measurement" operation.

    Mathematical Model:
        Nilpotent matrix N with N^k = 0.
        Information is progressively destroyed: after k applications,
        all input information is lost.

    Visual Effect:
        - Random noise added to all color channels
        - Information about original color is partially lost
        - Creates "static" or "grain" effect

    Jordan Normal Form:
        Corresponds to strictly upper triangular matrices (nilpotent).
        The MIX barrier: once crossed, original information is
        irrecoverable.

    Args:
        color: Input ColorState
        noise_level: Amplitude of noise injection [0, 1]
        seed: Optional random seed for reproducibility

    Returns:
        New ColorState with injected noise.
    """
    noise_level = clamp(noise_level, 0.0, 0.5)

    if seed is not None:
        random.seed(seed)

    # Add noise to each channel
    hue_noise = random.uniform(-1, 1) * noise_level * 60  # +/- 60 degrees max
    sat_noise = random.uniform(-1, 1) * noise_level
    lum_noise = random.uniform(-1, 1) * noise_level

    new_hue = normalize_hue(color.hue + hue_noise)
    new_sat = clamp(color.saturation + sat_noise, 0.0, 1.0)
    new_lum = clamp(color.luminance + lum_noise, 0.0, 1.0)

    # Coherence is reduced by noise (information destruction)
    new_coherence = max(0.0, color.coherence * (1.0 - noise_level))

    return ColorState(
        H=new_hue,
        S=new_sat,
        L=new_lum,
        alpha=color.alpha,
        coherence=new_coherence,
        gradient_rate=color.gradient_rate * (1.0 - noise_level)
    )


# =============================================================================
# Signature Analysis Functions
# =============================================================================

def signature_distance(sig1: PrimitiveSignature, sig2: PrimitiveSignature) -> float:
    """
    Compute the distance between two primitive signatures.

    Uses Euclidean distance in the normalized 6-dimensional signature space.
    This measures how different two color states are in terms of their
    primitive character.

    Args:
        sig1: First PrimitiveSignature
        sig2: Second PrimitiveSignature

    Returns:
        Distance in [0, sqrt(2)]. Zero means identical signatures.

    Interpretation:
        - Distance < 0.3: Very similar primitive character
        - Distance 0.3-0.7: Moderately different
        - Distance > 0.7: Very different primitive character
    """
    # Normalize both signatures for fair comparison
    n1 = sig1.normalize()
    n2 = sig2.normalize()

    # Euclidean distance
    diff_fix = n1.fix - n2.fix
    diff_repel = n1.repel - n2.repel
    diff_inv = n1.inv - n2.inv
    diff_osc = n1.osc - n2.osc
    diff_halt = n1.halt - n2.halt
    diff_mix = n1.mix - n2.mix

    return math.sqrt(
        diff_fix ** 2 +
        diff_repel ** 2 +
        diff_inv ** 2 +
        diff_osc ** 2 +
        diff_halt ** 2 +
        diff_mix ** 2
    )


def blend_signatures(
    sig1: PrimitiveSignature,
    sig2: PrimitiveSignature,
    t: float = 0.5
) -> PrimitiveSignature:
    """
    Blend two primitive signatures.

    Linearly interpolates between two signatures, useful for smooth
    transitions between color states with different primitive characters.

    Args:
        sig1: First PrimitiveSignature (weight 1-t)
        sig2: Second PrimitiveSignature (weight t)
        t: Blend parameter in [0, 1]. t=0 gives sig1, t=1 gives sig2.

    Returns:
        New PrimitiveSignature blended between the two inputs.
    """
    t = clamp(t, 0.0, 1.0)
    s = 1.0 - t

    return PrimitiveSignature(
        fix=s * sig1.fix + t * sig2.fix,
        repel=s * sig1.repel + t * sig2.repel,
        inv=s * sig1.inv + t * sig2.inv,
        osc=s * sig1.osc + t * sig2.osc,
        halt=s * sig1.halt + t * sig2.halt,
        mix=s * sig1.mix + t * sig2.mix
    )


def signature_entropy(sig: PrimitiveSignature) -> float:
    """
    Compute the Shannon entropy of a primitive signature.

    High entropy indicates the color state has roughly equal affinity
    for all primitives (generalist). Low entropy indicates dominance
    by one or two primitives (specialist).

    Args:
        sig: PrimitiveSignature to analyze

    Returns:
        Entropy in bits. Maximum is log2(6) = 2.58 for uniform distribution.
    """
    norm = sig.normalize()
    values = [norm.fix, norm.repel, norm.inv, norm.osc, norm.halt, norm.mix]

    entropy = 0.0
    for p in values:
        if p > 0:
            entropy -= p * math.log2(p)

    return entropy


def classify_by_signature(sig: PrimitiveSignature) -> str:
    """
    Classify a color state by its signature characteristics.

    Returns a human-readable classification based on the signature's
    dominant components and overall character.

    Args:
        sig: PrimitiveSignature to classify

    Returns:
        Classification string describing the color's character.
    """
    norm = sig.normalize()
    dominant = norm.dominant()
    entropy = signature_entropy(sig)

    # Determine overall character
    diag_content = norm.diagonalizable_content()
    jordan_content = norm.jordan_content()

    if entropy > 2.0:
        character = "balanced"
    elif dominant in ['FIX', 'REPEL']:
        character = "stable" if dominant == 'FIX' else "dynamic"
    elif dominant == 'INV':
        character = "rotating"
    elif dominant == 'OSC':
        character = "oscillating"
    elif dominant == 'HALT':
        character = "critical"
    else:  # MIX
        character = "chaotic"

    # Quantum/classical indicator
    if diag_content > 0.8:
        realm = "quantum-like"
    elif jordan_content > 0.5:
        realm = "classical-like"
    else:
        realm = "boundary"

    return f"{character} ({realm}, dominant: {dominant})"


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Constants
    'TAU_FIX',
    'GOLDEN_ANGLE',
    'DEFAULT_NOISE_LEVEL',
    'HALT_BANDS',

    # Core dataclass
    'PrimitiveSignature',

    # Signature computation
    'compute_color_signature',
    'compute_signature_from_color',
    'hsl_to_signature',

    # Primitive operations
    'apply_fix',
    'apply_repel',
    'apply_inv',
    'apply_osc',
    'apply_halt',
    'apply_mix',

    # Signature analysis
    'signature_distance',
    'blend_signatures',
    'signature_entropy',
    'classify_by_signature',
]
