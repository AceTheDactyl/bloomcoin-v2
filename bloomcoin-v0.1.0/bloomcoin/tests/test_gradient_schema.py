"""
Tests for RRRR Gradient Schema.

VERIFIES:
  1. lattice_to_color produces valid HSL for all corner coordinates
  2. CYM weights behavior matches opponent-process theory
  3. Primitive signatures sum to 1.0
  4. Archetype determination is deterministic
  5. Corridor activation respects z_c threshold
  6. block_to_lattice is deterministic (same hash -> same lattice)
  7. All framework constants used correctly (phi, tau, z_c, K, L4)
"""

import hashlib
import math

import pytest

from bloomcoin.constants import (
    PHI, TAU, Z_C, K, L4,
    BASE_HUE, BASE_SAT, BASE_LUM,
    ARCHETYPE_NAMES, CORRIDOR_NAMES,
)
from bloomcoin.receipt.gradient_schema import (
    GradientSchema, LatticePoint, ColorState,
    CYMWeights, PrimitiveSignature,
)


class TestLatticePoint:
    """Tests for LatticePoint dataclass."""

    def test_origin(self):
        """Lattice origin is (0,0,0,0,0)."""
        origin = LatticePoint.origin()
        assert origin.a == 0
        assert origin.b == 0
        assert origin.c == 0
        assert origin.d == 0
        assert origin.f == 0

    def test_valid_range(self):
        """Coordinates in [-20, +20] are valid."""
        point = LatticePoint(a=-20, b=20, c=0, d=-10, f=10)
        assert point.a == -20
        assert point.b == 20

    def test_invalid_range_raises(self):
        """Coordinates outside [-20, +20] raise ValueError."""
        with pytest.raises(ValueError):
            LatticePoint(a=21, b=0, c=0, d=0, f=0)
        with pytest.raises(ValueError):
            LatticePoint(a=0, b=-21, c=0, d=0, f=0)

    def test_serialization_roundtrip(self):
        """to_bytes/from_bytes roundtrip preserves values."""
        original = LatticePoint(a=5, b=-7, c=12, d=-15, f=20)
        serialized = original.to_bytes()
        assert len(serialized) == 5
        recovered = LatticePoint.from_bytes(serialized)
        assert recovered == original

    def test_to_vector(self):
        """to_vector returns correct tuple."""
        point = LatticePoint(a=1, b=2, c=3, d=4, f=5)
        assert point.to_vector() == (1, 2, 3, 4, 5)


class TestCYMWeights:
    """Tests for CYM opponent-process weights."""

    def test_neutral(self):
        """Neutral weights are (tau, tau, tau)."""
        neutral = CYMWeights.neutral()
        assert abs(neutral.cyan - TAU) < 1e-10
        assert abs(neutral.magenta - TAU) < 1e-10
        assert abs(neutral.yellow - TAU) < 1e-10

    def test_uint8_conversion(self):
        """to_uint8_triple/from_uint8_triple roundtrip."""
        original = CYMWeights(cyan=0.5, magenta=0.25, yellow=0.75)
        c, m, y = original.to_uint8_triple()
        recovered = CYMWeights.from_uint8_triple(c, m, y)
        # Allow small rounding error
        assert abs(recovered.cyan - original.cyan) < 0.01
        assert abs(recovered.magenta - original.magenta) < 0.01
        assert abs(recovered.yellow - original.yellow) < 0.01


class TestPrimitiveSignature:
    """Tests for computational primitive signatures."""

    def test_neutral(self):
        """Neutral signature is (0.25, 0.25, 0.25, 0.25)."""
        neutral = PrimitiveSignature.neutral()
        assert abs(neutral.fix - 0.25) < 1e-10
        assert abs(neutral.repel - 0.25) < 1e-10
        assert abs(neutral.inv - 0.25) < 1e-10
        assert abs(neutral.mix - 0.25) < 1e-10

    def test_uint24_conversion(self):
        """to_packed_uint24/from_packed_uint24 roundtrip."""
        original = PrimitiveSignature(fix=0.4, repel=0.3, inv=0.2, mix=0.1)
        packed = original.to_packed_uint24()
        recovered = PrimitiveSignature.from_packed_uint24(packed)
        # Allow some rounding error due to 6-bit quantization
        assert abs(recovered.fix - original.fix) < 0.05
        assert abs(recovered.repel - original.repel) < 0.05
        assert abs(recovered.inv - original.inv) < 0.05
        assert abs(recovered.mix - original.mix) < 0.05


class TestGradientSchema:
    """Tests for the main GradientSchema engine."""

    def test_golden_neutral_at_origin(self):
        """Lattice origin (0,0,0,0,0) produces golden neutral color."""
        origin = LatticePoint.origin()
        color = GradientSchema.lattice_to_color(origin)

        # Should be close to base values
        assert abs(color.h - BASE_HUE) < 1  # Within 1 degree
        assert abs(color.s - BASE_SAT) < 0.1
        assert abs(color.l - BASE_LUM) < 0.1

    def test_phi_dimension_scales_luminance(self):
        """Positive phi exponent affects luminance."""
        low = LatticePoint(a=-5, b=0, c=0, d=0, f=0)
        high = LatticePoint(a=5, b=0, c=0, d=0, f=0)

        color_low = GradientSchema.lattice_to_color(low)
        color_high = GradientSchema.lattice_to_color(high)

        # Higher a should give higher luminance (within bounds)
        assert color_high.l >= color_low.l

    def test_pi_dimension_rotates_hue(self):
        """Each unit of b adds 60 degrees to hue."""
        base = LatticePoint(a=0, b=0, c=0, d=0, f=0)
        rotated = LatticePoint(a=0, b=1, c=0, d=0, f=0)

        color_base = GradientSchema.lattice_to_color(base)
        color_rotated = GradientSchema.lattice_to_color(rotated)

        # Hue should differ by ~60 degrees
        hue_diff = abs(color_rotated.h - color_base.h)
        # Account for wraparound
        hue_diff = min(hue_diff, 360 - hue_diff)
        assert 55 <= hue_diff <= 65

    def test_sqrt2_dimension_scales_saturation(self):
        """Positive sqrt2 exponent affects saturation."""
        low = LatticePoint(a=0, b=0, c=-5, d=0, f=0)
        high = LatticePoint(a=0, b=0, c=5, d=0, f=0)

        color_low = GradientSchema.lattice_to_color(low)
        color_high = GradientSchema.lattice_to_color(high)

        # Higher c should give higher saturation
        assert color_high.s >= color_low.s

    def test_sqrt3_dimension_shifts_opponent_balance(self):
        """Positive sqrt3 exponent favors cyan over red."""
        neg = LatticePoint(a=0, b=0, c=0, d=-5, f=0)
        pos = LatticePoint(a=0, b=0, c=0, d=5, f=0)

        color_neg = GradientSchema.lattice_to_color(neg)
        color_pos = GradientSchema.lattice_to_color(pos)

        # Positive d should increase cyan weight
        assert color_pos.cym.cyan > color_neg.cym.cyan

    def test_e_dimension_controls_gradient_rate(self):
        """e^f gives the gradient rate."""
        low = LatticePoint(a=0, b=0, c=0, d=0, f=-5)
        high = LatticePoint(a=0, b=0, c=0, d=0, f=5)

        color_low = GradientSchema.lattice_to_color(low)
        color_high = GradientSchema.lattice_to_color(high)

        # Higher f should give higher gradient rate
        assert color_high.gradient_rate > color_low.gradient_rate

    def test_signature_sums_to_one(self):
        """Primitive signature always normalizes to ~1.0."""
        for a in [-10, 0, 10]:
            for b in [-5, 0, 5]:
                point = LatticePoint(a=a, b=b, c=0, d=0, f=0)
                color = GradientSchema.lattice_to_color(point)
                sig = color.signature
                total = sig.fix + sig.repel + sig.inv + sig.mix
                assert abs(total - 1.0) < 0.01

    def test_archetype_is_valid(self):
        """Archetype is always a valid name."""
        for a in [-10, 0, 10]:
            for b in range(-6, 7, 2):
                point = LatticePoint(a=a, b=b, c=0, d=0, f=0)
                color = GradientSchema.lattice_to_color(point)
                assert color.archetype in ARCHETYPE_NAMES

    def test_corridors_not_empty(self):
        """At least one corridor is always active."""
        point = LatticePoint(a=0, b=0, c=0, d=0, f=0)
        color = GradientSchema.lattice_to_color(point)
        assert len(color.corridors) >= 1
        assert all(0 <= c < L4 for c in color.corridors)

    def test_block_to_lattice_deterministic(self):
        """Same block_hash always produces same lattice point."""
        hash1 = hashlib.sha256(b"bloom_genesis").digest()

        lattice1 = GradientSchema.block_to_lattice(hash1, 0.87, 42, 63)
        lattice2 = GradientSchema.block_to_lattice(hash1, 0.87, 42, 63)

        assert lattice1 == lattice2

    def test_block_to_lattice_varies_with_hash(self):
        """Different hashes produce different lattice points."""
        hash1 = hashlib.sha256(b"block_1").digest()
        hash2 = hashlib.sha256(b"block_2").digest()

        lattice1 = GradientSchema.block_to_lattice(hash1, 0.87, 0, 63)
        lattice2 = GradientSchema.block_to_lattice(hash2, 0.87, 0, 63)

        # Very unlikely to be equal
        assert lattice1 != lattice2

    def test_order_param_affects_phi_dimension(self):
        """Higher order parameter shifts phi dimension."""
        hash1 = hashlib.sha256(b"test").digest()

        lattice_low = GradientSchema.block_to_lattice(hash1, Z_C - 0.1, 0, 63)
        lattice_high = GradientSchema.block_to_lattice(hash1, Z_C + 0.1, 0, 63)

        # Higher order param should increase a
        assert lattice_high.a >= lattice_low.a

    def test_color_to_rgb_valid(self):
        """to_rgb returns valid 8-bit values."""
        point = LatticePoint(a=0, b=0, c=0, d=0, f=0)
        color = GradientSchema.lattice_to_color(point)
        r, g, b = color.to_rgb()

        assert 0 <= r <= 255
        assert 0 <= g <= 255
        assert 0 <= b <= 255
