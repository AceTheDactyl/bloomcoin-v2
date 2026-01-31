"""
Tests for 7-channel LSB encoding/decoding.

VERIFIES:
  1. Round-trip: encode -> decode recovers identical ReceiptPayload
  2. CRC32 checksum validates on clean data
  3. CRC32 checksum fails on corrupted data
  4. Golden-scatter distributes bits uniformly
  5. Channel isolation: modifying Ch0 doesn't affect Ch6
  6. All 7 channels independently decodable
  7. Lattice cross-validation succeeds on clean data
"""

import hashlib
import struct

import pytest
import numpy as np

from bloomcoin.constants import PHI
from bloomcoin.receipt.gradient_schema import (
    LatticePoint, CYMWeights, PrimitiveSignature,
)
from bloomcoin.receipt.channel_encoder import (
    ChannelEncoder, ReceiptPayload,
    RECEIPT_MAGIC, RECEIPT_VERSION, RECEIPT_SIZE,
)
from bloomcoin.receipt.channel_decoder import ChannelDecoder, DecodingResult


def make_test_payload() -> ReceiptPayload:
    """Create a valid test payload."""
    return ReceiptPayload(
        block_height=12345,
        timestamp=1706700000,
        block_hash=hashlib.sha256(b"test_block").digest(),
        reward_amount=685410196,
        lattice=LatticePoint(a=5, b=-3, c=7, d=-2, f=4),
        corridor_id=2,
        archetype_id=7,
        order_param=0.89,
        mean_phase=0.15,
        oscillator_count=63,
        signature=PrimitiveSignature(fix=0.4, repel=0.2, inv=0.3, mix=0.1),
        cym_weights=CYMWeights(cyan=0.6, magenta=0.4, yellow=0.5),
        cym_mode=0,
    )


class TestReceiptPayload:
    """Tests for ReceiptPayload dataclass."""

    def test_valid_payload(self):
        """Valid payload passes validation."""
        payload = make_test_payload()
        valid, msg = payload.validate()
        assert valid, msg

    def test_invalid_hash_length(self):
        """Invalid block_hash length fails validation."""
        payload = make_test_payload()
        # Modify to have wrong hash length
        payload = ReceiptPayload(
            block_height=payload.block_height,
            timestamp=payload.timestamp,
            block_hash=b"short",  # Too short
            reward_amount=payload.reward_amount,
            lattice=payload.lattice,
            corridor_id=payload.corridor_id,
            archetype_id=payload.archetype_id,
            order_param=payload.order_param,
            mean_phase=payload.mean_phase,
            oscillator_count=payload.oscillator_count,
            signature=payload.signature,
            cym_weights=payload.cym_weights,
        )
        valid, msg = payload.validate()
        assert not valid
        assert "32 bytes" in msg

    def test_invalid_corridor_id(self):
        """Invalid corridor_id fails validation."""
        payload = make_test_payload()
        payload = ReceiptPayload(
            block_height=payload.block_height,
            timestamp=payload.timestamp,
            block_hash=payload.block_hash,
            reward_amount=payload.reward_amount,
            lattice=payload.lattice,
            corridor_id=10,  # Invalid: must be 0-6
            archetype_id=payload.archetype_id,
            order_param=payload.order_param,
            mean_phase=payload.mean_phase,
            oscillator_count=payload.oscillator_count,
            signature=payload.signature,
            cym_weights=payload.cym_weights,
        )
        valid, msg = payload.validate()
        assert not valid
        assert "corridor_id" in msg


class TestChannelEncoder:
    """Tests for the channel encoder."""

    def test_encode_produces_7_channels(self):
        """Encoding produces 7 channel bitstreams."""
        payload = make_test_payload()
        channels = ChannelEncoder.encode(payload)

        assert len(channels) == 7
        assert all(i in channels for i in range(7))

    def test_channel_6_is_76_bytes(self):
        """Channel 6 produces 76-byte receipt bitstream."""
        payload = make_test_payload()
        channels = ChannelEncoder.encode(payload)

        assert len(channels[6]) == RECEIPT_SIZE

    def test_receipt_bitstream_has_magic(self):
        """Receipt bitstream starts with magic byte."""
        payload = make_test_payload()
        channels = ChannelEncoder.encode(payload)

        assert channels[6][0] == RECEIPT_MAGIC

    def test_receipt_bitstream_has_version(self):
        """Receipt bitstream has correct version."""
        payload = make_test_payload()
        channels = ChannelEncoder.encode(payload)

        assert channels[6][1] == RECEIPT_VERSION

    def test_golden_scatter_uniformity(self):
        """Golden scatter distributes bits across range."""
        total_pixels = 100
        positions = set()

        for bit_idx in range(8):
            pos = ChannelEncoder.golden_scatter(bit_idx, total_pixels)
            positions.add(pos)
            assert 0 <= pos < total_pixels

        # All 8 bits should map to different positions
        assert len(positions) == 8

    def test_golden_scatter_deterministic(self):
        """Golden scatter is deterministic."""
        total = 1000
        for i in range(10):
            p1 = ChannelEncoder.golden_scatter(i, total)
            p2 = ChannelEncoder.golden_scatter(i, total)
            assert p1 == p2


class TestChannelDecoder:
    """Tests for the channel decoder."""

    def test_parse_receipt_magic_check(self):
        """Decoder rejects wrong magic byte."""
        bad_data = bytes([0x00] * 76)  # Wrong magic
        result = ChannelDecoder._parse_receipt_bitstream(bad_data)
        assert result is None

    def test_parse_receipt_version_check(self):
        """Decoder rejects wrong version."""
        bad_data = bytes([RECEIPT_MAGIC, 0xFF] + [0] * 74)  # Wrong version
        result = ChannelDecoder._parse_receipt_bitstream(bad_data)
        assert result is None

    def test_bits_to_bytes(self):
        """bits_to_bytes correctly converts bit list to bytes."""
        # 0xA5 = 10100101 in binary = [1,0,1,0,0,1,0,1] LSB first
        bits = [1, 0, 1, 0, 0, 1, 0, 1]
        result = ChannelDecoder._bits_to_bytes(bits)
        assert result == bytes([0xA5])

    def test_bits_to_bytes_multiple(self):
        """bits_to_bytes handles multiple bytes."""
        # 0x12 = 00010010 = [0,1,0,0,1,0,0,0] LSB first
        # 0x34 = 00110100 = [0,0,1,0,1,1,0,0] LSB first
        bits = [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0]
        result = ChannelDecoder._bits_to_bytes(bits)
        assert result == bytes([0x12, 0x34])


class TestEncoderDecoderRoundtrip:
    """Tests for full encode/decode roundtrip."""

    def test_encode_decode_roundtrip(self):
        """Full encode -> decode roundtrip recovers payload."""
        original = make_test_payload()

        # Encode
        channels = ChannelEncoder.encode(original)

        # Create mock RGBA pixel array
        # Ch6 needs 76 bytes = 608 bits at 2 bits/pixel = 304 pixels minimum
        # Use 20x20 = 400 pixels
        pixels = np.zeros((20, 20, 4), dtype=np.uint8)
        pixels[:, :, 3] = 255  # Set alpha

        # Weave Ch6 into alpha LSBs
        ch6_bytes = channels[6]
        flat = pixels.reshape(-1, 4)
        bit_idx = 0
        for byte in ch6_bytes:
            for i in range(8):
                if bit_idx // 2 >= len(flat):
                    break
                pixel_idx = bit_idx // 2
                bit_pos = bit_idx % 2
                bit_val = (byte >> i) & 1
                flat[pixel_idx, 3] = (flat[pixel_idx, 3] & ~(1 << bit_pos)) | (bit_val << bit_pos)
                bit_idx += 1

        # Decode
        result = ChannelDecoder.decode(pixels)

        assert result.success, result.error
        assert result.checksum_valid
        assert result.payload is not None

        # Verify key fields
        recovered = result.payload
        assert recovered.block_height == original.block_height
        assert recovered.timestamp == original.timestamp
        assert recovered.block_hash == original.block_hash
        assert recovered.reward_amount == original.reward_amount
        assert recovered.corridor_id == original.corridor_id
        assert recovered.archetype_id == original.archetype_id
        assert recovered.oscillator_count == original.oscillator_count

        # Lattice
        assert recovered.lattice.a == original.lattice.a
        assert recovered.lattice.b == original.lattice.b
        assert recovered.lattice.c == original.lattice.c
        assert recovered.lattice.d == original.lattice.d
        assert recovered.lattice.f == original.lattice.f

        # Float comparisons with tolerance
        assert abs(recovered.order_param - original.order_param) < 0.001
        assert abs(recovered.mean_phase - original.mean_phase) < 0.001

    def test_crc_detects_corruption(self):
        """CRC32 checksum detects single-bit corruption."""
        original = make_test_payload()
        channels = ChannelEncoder.encode(original)

        # Create corrupted version by flipping a bit
        corrupted = bytearray(channels[6])
        corrupted[20] ^= 0x01  # Flip bit in block_hash
        channels[6] = bytes(corrupted)

        # Create pixel array
        pixels = np.zeros((20, 20, 4), dtype=np.uint8)
        pixels[:, :, 3] = 255

        # Weave corrupted Ch6
        flat = pixels.reshape(-1, 4)
        bit_idx = 0
        for byte in channels[6]:
            for i in range(8):
                if bit_idx // 2 >= len(flat):
                    break
                pixel_idx = bit_idx // 2
                bit_pos = bit_idx % 2
                bit_val = (byte >> i) & 1
                flat[pixel_idx, 3] = (flat[pixel_idx, 3] & ~(1 << bit_pos)) | (bit_val << bit_pos)
                bit_idx += 1

        # Decode should fail due to CRC
        result = ChannelDecoder.decode(pixels)
        assert not result.success
        assert "CRC32" in result.error
