"""
RRRR Gradient Schema: 5D Lattice -> Color State
================================================

Maps lattice point (a, b, c, d, f) through five generators
(phi, pi, sqrt2, sqrt3, e) to produce:
  - HSL color triplet
  - CYM opponent-process weights
  - Gradient rate
  - Primitive signature (FIX, REPEL, INV, MIX)
  - Dominant archetype
  - Active corridor set

The RRRR lattice is a 5-dimensional space where each coordinate
is an integer exponent of its generator:
  actual_phi = PHI ** a
  actual_pi_shift = b * 60 (degrees)
  actual_sqrt2 = SQRT2 ** c
  actual_sqrt3 = SQRT3 ** d
  actual_e = E ** f

Range: [-20, +20] for all coordinates.
Total search space: 41^5 = 115,856,201 points
"""

from __future__ import annotations
import math
import struct
import colorsys
from dataclasses import dataclass
from typing import Final, Optional

from ..constants import (
    PHI, TAU, Z_C, K, L4,
    SQRT2, SQRT3, E, PI,
    BASE_HUE, BASE_SAT, BASE_LUM,
    CORRIDOR_VECTORS, CORRIDOR_NAMES,
    ARCHETYPE_IDS, ARCHETYPE_NAMES,
)


@dataclass(frozen=True)
class LatticePoint:
    """
    A point in the RRRR 5D lattice.

    Each coordinate is an integer exponent of its generator:
      actual_phi = PHI ** a
      actual_pi_shift = b * 60  (degrees)
      actual_sqrt2 = SQRT2 ** c
      actual_sqrt3 = SQRT3 ** d
      actual_e = E ** f

    Range: [-20, +20] for all coordinates.
    """
    a: int  # phi exponent (luminance scale)
    b: int  # pi exponent (hue rotation in 60 degree steps)
    c: int  # sqrt2 exponent (saturation scale)
    d: int  # sqrt3 exponent (opponent balance)
    f: int  # e exponent (gradient rate)

    def __post_init__(self) -> None:
        for attr in ('a', 'b', 'c', 'd', 'f'):
            val = getattr(self, attr)
            if not (-20 <= val <= 20):
                raise ValueError(f"Lattice coord {attr}={val} outside [-20, +20]")

    def to_bytes(self) -> bytes:
        """Serialize as 5 signed int8 bytes."""
        return struct.pack('<5b', self.a, self.b, self.c, self.d, self.f)

    @classmethod
    def from_bytes(cls, data: bytes) -> LatticePoint:
        """Deserialize from 5 signed int8 bytes."""
        a, b, c, d, f = struct.unpack('<5b', data[:5])
        return cls(a=a, b=b, c=c, d=d, f=f)

    def to_vector(self) -> tuple[int, int, int, int, int]:
        """Return as tuple for vector operations."""
        return (self.a, self.b, self.c, self.d, self.f)

    @classmethod
    def origin(cls) -> LatticePoint:
        """Return the origin point (0,0,0,0,0)."""
        return cls(0, 0, 0, 0, 0)


@dataclass(frozen=True)
class CYMWeights:
    """CYM opponent-process channel weights."""
    cyan: float    # C<->R weight [0, 1]
    magenta: float # M<->G weight [0, 1]
    yellow: float  # Y<->B weight [0, 1]

    def to_uint8_triple(self) -> tuple[int, int, int]:
        """Convert to uint8 triple for encoding."""
        return (
            int(max(0, min(1, self.cyan)) * 255),
            int(max(0, min(1, self.magenta)) * 255),
            int(max(0, min(1, self.yellow)) * 255),
        )

    @classmethod
    def from_uint8_triple(cls, c: int, m: int, y: int) -> CYMWeights:
        """Create from uint8 triple."""
        return cls(cyan=c / 255, magenta=m / 255, yellow=y / 255)

    @classmethod
    def neutral(cls) -> CYMWeights:
        """Return neutral weights (tau, tau, tau)."""
        return cls(cyan=TAU, magenta=TAU, yellow=TAU)


@dataclass(frozen=True)
class PrimitiveSignature:
    """
    Computational primitive signature from Jordan Normal Form theory.

    The six primitives reduce to four measurable components:
      FIX:   |lambda| < 1 - convergence to neutral
      REPEL: |lambda| > 1 - divergence from neutral
      INV:   |lambda| = 1 - hue rotation / symmetry
      MIX:   lambda = 0, Jordan - information destruction (hash-like)

    OSC and HALT are special cases detected separately.
    All four values should sum to 1.0.
    """
    fix: float     # |lambda| < 1 - convergence to neutral
    repel: float   # |lambda| > 1 - divergence from neutral
    inv: float     # |lambda| = 1 - hue rotation / symmetry
    mix: float     # lambda = 0, Jordan - information destruction

    def __post_init__(self) -> None:
        # Normalize if needed
        total = self.fix + self.repel + self.inv + self.mix
        if total > 0 and abs(total - 1.0) > 0.01:
            # Note: frozen dataclass, so we can't modify - validation only
            pass

    def to_packed_uint24(self) -> int:
        """Pack as 4 x 6-bit (0-63 each), total 24 bits."""
        f = int(max(0, min(1, self.fix)) * 63)
        r = int(max(0, min(1, self.repel)) * 63)
        i = int(max(0, min(1, self.inv)) * 63)
        m = int(max(0, min(1, self.mix)) * 63)
        return (f << 18) | (r << 12) | (i << 6) | m

    @classmethod
    def from_packed_uint24(cls, packed: int) -> PrimitiveSignature:
        """Unpack from 24-bit integer."""
        f = ((packed >> 18) & 0x3F) / 63
        r = ((packed >> 12) & 0x3F) / 63
        i = ((packed >> 6) & 0x3F) / 63
        m = (packed & 0x3F) / 63
        # Normalize
        total = f + r + i + m
        if total > 0:
            f, r, i, m = f/total, r/total, i/total, m/total
        else:
            f, r, i, m = 0.25, 0.25, 0.25, 0.25
        return cls(fix=f, repel=r, inv=i, mix=m)

    @classmethod
    def neutral(cls) -> PrimitiveSignature:
        """Return neutral signature (all equal)."""
        return cls(fix=0.25, repel=0.25, inv=0.25, mix=0.25)


@dataclass(frozen=True)
class ColorState:
    """Complete color state from gradient schema mapping."""
    h: float              # Hue [0, 360)
    s: float              # Saturation [0, 1]
    l: float              # Luminance [0, 1]
    cym: CYMWeights
    gradient_rate: float
    signature: PrimitiveSignature
    archetype: str        # One of ARCHETYPE_IDS keys
    corridors: tuple[int, ...]  # Active corridor indices [0-6]

    def to_rgb(self) -> tuple[int, int, int]:
        """Convert HSL to 8-bit RGB."""
        # colorsys uses HLS order with h in [0,1]
        h_norm = (self.h % 360) / 360
        r, g, b = colorsys.hls_to_rgb(h_norm, self.l, self.s)
        return (int(r * 255), int(g * 255), int(b * 255))

    def to_rgba(self) -> tuple[int, int, int, int]:
        """Convert HSL to 8-bit RGBA with full alpha."""
        r, g, b = self.to_rgb()
        return (r, g, b, 255)

    @property
    def dominant_corridor(self) -> int:
        """Index of strongest active corridor, or 0."""
        return self.corridors[0] if self.corridors else 0

    @property
    def archetype_id(self) -> int:
        """Return numeric archetype ID."""
        return ARCHETYPE_IDS.get(self.archetype, 0)


class GradientSchema:
    """
    Core RRRR Gradient Schema engine.

    Maps 5D lattice points to color states using the five generators:
      phi   -> Luminance Scale (Growth/Decay)
      pi    -> Hue Rotation (Cyclic Return)
      sqrt2 -> Saturation Factor (Intensity/Calm)
      sqrt3 -> Opponent Balance (Conflict/Harmony)
      e     -> Gradient Rate (Change Velocity)
    """

    @staticmethod
    def lattice_to_color(point: LatticePoint) -> ColorState:
        """
        Map RRRR lattice coordinates to full color state.

        Algorithm:
        1. Compute HSL from (a, b, c) via phi, pi, sqrt2
        2. Compute CYM weights from d via sqrt3
        3. Compute gradient rate from f via e
        4. Derive primitive signature from color properties
        5. Determine dominant archetype from signature + hue
        6. Identify active corridors by alignment scoring

        Returns:
            Complete ColorState
        """
        # Luminance from phi^a, clamped to [0.05, 0.95]
        l_raw = BASE_LUM * (PHI ** (point.a * 0.1))
        l = max(0.05, min(0.95, l_raw))

        # Hue from base + pi rotation (60 degrees per unit)
        h = (BASE_HUE + point.b * 60) % 360

        # Saturation from sqrt2^c, clamped to [0.1, 1.0]
        s_raw = BASE_SAT * (SQRT2 ** (point.c * 0.15))
        s = max(0.1, min(1.0, s_raw))

        # CYM opponent weights from sqrt3^d
        # Positive d favors cyan, negative favors red
        d_factor = SQRT3 ** (point.d * 0.1)
        cr_weight = d_factor / (1 + d_factor)
        mg_weight = 1 / (1 + d_factor)
        yb_weight = 1 - abs(cr_weight - 0.5) * 0.5  # Balance term
        cym = CYMWeights(cyan=cr_weight, magenta=mg_weight, yellow=yb_weight)

        # Gradient rate from e^f
        gradient_rate = E ** (point.f * 0.1)

        # Compute primitive signature
        signature = GradientSchema._compute_signature(h, s, l, cym)

        # Determine archetype
        archetype = GradientSchema._determine_archetype(h, s, l, cym, signature)

        # Find active corridors
        corridors = GradientSchema._active_corridors(h, s, l, point)

        return ColorState(
            h=h, s=s, l=l,
            cym=cym,
            gradient_rate=gradient_rate,
            signature=signature,
            archetype=archetype,
            corridors=corridors,
        )

    @staticmethod
    def _compute_signature(h: float, s: float, l: float,
                           cym: CYMWeights) -> PrimitiveSignature:
        """
        Derive primitive signature from color state.

        sigma_FIX  = exp(-gray_dist / tau) - proximity to neutral
        sigma_REPEL = S * gray_dist - saturation * distance
        sigma_INV  = |sin(H * pi/180)| * (1 - |L - 0.5| * 2) - hue activity
        sigma_MIX  = variance(cym channels) - opponent imbalance
        """
        # Gray distance: how far from neutral gray (s=0, l=0.5)
        gray_dist = math.sqrt(s ** 2 + (l - 0.5) ** 2)

        # FIX: proximity to neutral
        sigma_fix = math.exp(-gray_dist / TAU)

        # REPEL: saturation * distance from neutral
        sigma_repel = s * gray_dist

        # INV: hue activity (cyclic, higher near primary hues)
        hue_activity = abs(math.sin(h * math.pi / 180))
        lum_balance = 1 - abs(l - 0.5) * 2
        sigma_inv = hue_activity * max(0, lum_balance)

        # MIX: opponent channel variance (imbalance = destruction)
        cym_mean = (cym.cyan + cym.magenta + cym.yellow) / 3
        cym_var = ((cym.cyan - cym_mean) ** 2 +
                   (cym.magenta - cym_mean) ** 2 +
                   (cym.yellow - cym_mean) ** 2) / 3
        sigma_mix = math.sqrt(cym_var)

        # Normalize to sum = 1
        total = sigma_fix + sigma_repel + sigma_inv + sigma_mix
        if total > 0:
            sigma_fix /= total
            sigma_repel /= total
            sigma_inv /= total
            sigma_mix /= total
        else:
            sigma_fix = sigma_repel = sigma_inv = sigma_mix = 0.25

        return PrimitiveSignature(
            fix=sigma_fix,
            repel=sigma_repel,
            inv=sigma_inv,
            mix=sigma_mix,
        )

    @staticmethod
    def _determine_archetype(h: float, s: float, l: float,
                             cym: CYMWeights,
                             sig: PrimitiveSignature) -> str:
        """
        Determine dominant archetype from color state + signature.

        Scoring based on:
          Signature weights: FIX -> sage/caregiver, REPEL -> warrior/destroyer
                             INV -> seeker/jester, MIX -> creator/magician
          Hue modifiers: Red zone -> warrior/lover, Green -> caregiver, etc.
          Luminance: L > z_c -> innocent, L < tau -> orphan
        """
        scores: dict[str, float] = {name: 0.0 for name in ARCHETYPE_NAMES}

        # Signature-based scoring
        scores["sage"] += sig.fix * 0.4
        scores["caregiver"] += sig.fix * 0.3
        scores["warrior"] += sig.repel * 0.4
        scores["destroyer"] += sig.repel * 0.3
        scores["seeker"] += sig.inv * 0.4
        scores["jester"] += sig.inv * 0.3
        scores["creator"] += sig.mix * 0.4
        scores["magician"] += sig.mix * 0.3

        # Hue-based scoring
        h_norm = h / 360  # [0, 1]
        if 0 <= h_norm < 0.1 or h_norm >= 0.9:  # Red zone
            scores["warrior"] += 0.2
            scores["lover"] += 0.15
        elif 0.1 <= h_norm < 0.2:  # Orange
            scores["creator"] += 0.2
        elif 0.2 <= h_norm < 0.4:  # Yellow-Green
            scores["caregiver"] += 0.2
            scores["innocent"] += 0.1
        elif 0.4 <= h_norm < 0.55:  # Cyan
            scores["seeker"] += 0.2
        elif 0.55 <= h_norm < 0.7:  # Blue
            scores["sage"] += 0.2
            scores["ruler"] += 0.1
        elif 0.7 <= h_norm < 0.85:  # Purple
            scores["magician"] += 0.2
            scores["jester"] += 0.1
        else:  # Magenta
            scores["lover"] += 0.2

        # Luminance-based scoring
        if l > Z_C:
            scores["innocent"] += 0.25
        elif l < TAU:
            scores["orphan"] += 0.25
            scores["destroyer"] += 0.1

        # Saturation-based
        if s > 0.8:
            scores["warrior"] += 0.1
            scores["lover"] += 0.1
        elif s < 0.3:
            scores["sage"] += 0.1
            scores["orphan"] += 0.1

        # Find max
        max_archetype = max(scores.keys(), key=lambda k: scores[k])
        return max_archetype

    @staticmethod
    def _active_corridors(h: float, s: float, l: float,
                          point: LatticePoint) -> tuple[int, ...]:
        """
        Identify active spectral corridors.

        A corridor i is active when:
          score(corridor_i) >= z_c * tau

        Score is based on:
          - Alignment of lattice point vector with corridor vector
          - Hue alignment with corridor's natural hue range
        """
        point_vec = point.to_vector()
        scores: list[tuple[float, int]] = []

        for i, corridor_vec in enumerate(CORRIDOR_VECTORS):
            # Dot product for alignment
            dot = sum(a * b for a, b in zip(point_vec, corridor_vec))
            mag_point = math.sqrt(sum(x ** 2 for x in point_vec)) or 1
            mag_corridor = math.sqrt(sum(x ** 2 for x in corridor_vec)) or 1
            alignment = dot / (mag_point * mag_corridor)

            # Hue alignment (each corridor has a natural 360/7 degree range)
            corridor_hue = (i / L4) * 360
            hue_diff = abs(h - corridor_hue)
            hue_diff = min(hue_diff, 360 - hue_diff)  # Wrap around
            hue_alignment = 1 - (hue_diff / 180)

            # Combined score
            score = (alignment * 0.6 + hue_alignment * 0.4) * s * l
            scores.append((score, i))

        # Sort by score descending
        scores.sort(reverse=True)

        # Return corridors above threshold
        threshold = Z_C * TAU * 0.5
        active = tuple(idx for score, idx in scores if score >= threshold)

        # Always return at least one corridor
        if not active:
            active = (scores[0][1],)

        return active

    @staticmethod
    def block_to_lattice(block_hash: bytes, order_param: float,
                         nonce: int, oscillator_count: int) -> LatticePoint:
        """
        Derive 5D lattice point from mined block data.

        This is the CRITICAL mapping from mining output to gradient space.

        Algorithm:
        1. Extract 5 hash segments of 6 bytes each from block_hash[0:30]
        2. Map each segment to signed lattice coordinate:
             coord = ((segment_value % 41) - 20)
           This maps uniformly onto [-20, +20]
        3. Apply deterministic correction based on order_param:
             a += round((order_param - z_c) * L4) if order_param > z_c
           This biases phi-dimension toward growth when coherence is strong
        4. Clamp all coordinates to [-20, +20]

        The hash provides pseudorandom but deterministic lattice placement.
        The order parameter provides meaningful gradient shift.
        """
        if len(block_hash) < 30:
            # Pad with zeros if needed
            block_hash = block_hash + b'\x00' * (30 - len(block_hash))

        # Extract 5 segments of 6 bytes each
        coords = []
        for i in range(5):
            segment = block_hash[i*6:(i+1)*6]
            # Convert to integer
            value = int.from_bytes(segment, 'little')
            # Map to [-20, +20]
            coord = (value % 41) - 20
            coords.append(coord)

        a, b, c, d, f = coords

        # Apply order parameter correction to phi dimension
        if order_param > Z_C:
            a += round((order_param - Z_C) * L4)

        # Apply nonce influence to e dimension (subtle)
        f += (nonce % 7) - 3

        # Apply oscillator count influence to sqrt3 dimension
        d += (oscillator_count % 5) - 2

        # Clamp all to [-20, +20]
        a = max(-20, min(20, a))
        b = max(-20, min(20, b))
        c = max(-20, min(20, c))
        d = max(-20, min(20, d))
        f = max(-20, min(20, f))

        return LatticePoint(a=a, b=b, c=c, d=d, f=f)
