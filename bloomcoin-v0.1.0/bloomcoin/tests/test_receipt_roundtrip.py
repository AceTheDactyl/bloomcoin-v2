"""
End-to-end receipt tests.

VERIFIES:
  1. Build receipt PNG from mock block data
  2. Verify receipt PNG produces matching payload
  3. Visual layer is visually reasonable (no black/white artifacts)
  4. File size is reasonable for 63x63 PNG
  5. Receipt survives PNG save/load cycle (lossless)
  6. Multiple receipts with different blocks produce different images
  7. Tampered image fails verification
"""

import hashlib
import io
import tempfile
from pathlib import Path

import pytest
import numpy as np

# Check if Pillow is available
try:
    from PIL import Image
    HAS_PILLOW = True
except ImportError:
    HAS_PILLOW = False

from bloomcoin.constants import (
    RECEIPT_SIZE_STANDARD, ARCHETYPE_NAMES, CORRIDOR_NAMES,
)


# Skip all tests if Pillow not installed
pytestmark = pytest.mark.skipif(not HAS_PILLOW, reason="Pillow not installed")


if HAS_PILLOW:
    from bloomcoin.receipt import ReceiptBuilder, ReceiptVerifier


class TestReceiptBuilder:
    """Tests for receipt building."""

    def test_build_creates_image(self):
        """Builder creates a PIL Image."""
        builder = ReceiptBuilder(size=63)

        mock_hash = hashlib.sha256(b"test_block_0").digest()
        receipt = builder.build(
            block_hash=mock_hash,
            block_height=1,
            timestamp=1706700000,
            reward_amount=685410196,
            order_param=0.89,
            mean_phase=0.15,
            oscillator_count=63,
            nonce=7,
        )

        assert receipt is not None
        assert receipt.mode == 'RGBA'
        assert receipt.size == (63, 63)

    def test_build_different_sizes(self):
        """Builder respects size parameter."""
        for size in [49, 63, 100]:
            builder = ReceiptBuilder(size=size)
            mock_hash = hashlib.sha256(b"test").digest()
            receipt = builder.build(
                block_hash=mock_hash,
                block_height=1,
                timestamp=1706700000,
                reward_amount=100000000,
                order_param=0.87,
                mean_phase=0.0,
                oscillator_count=63,
                nonce=0,
            )
            assert receipt.size == (size, size)

    def test_save_creates_file(self):
        """save() creates a PNG file."""
        builder = ReceiptBuilder(size=63)
        mock_hash = hashlib.sha256(b"test").digest()
        receipt = builder.build(
            block_hash=mock_hash,
            block_height=1,
            timestamp=1706700000,
            reward_amount=100000000,
            order_param=0.87,
            mean_phase=0.0,
            oscillator_count=63,
            nonce=0,
        )

        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            path = Path(f.name)

        try:
            builder.save(receipt, path)
            assert path.exists()
            assert path.stat().st_size > 0

            # Verify it's a valid PNG
            loaded = Image.open(path)
            assert loaded.mode == 'RGBA'
            assert loaded.size == (63, 63)
        finally:
            path.unlink(missing_ok=True)


class TestReceiptVerifier:
    """Tests for receipt verification."""

    def test_verify_valid_receipt(self):
        """Verifier correctly validates a valid receipt."""
        builder = ReceiptBuilder(size=63)
        verifier = ReceiptVerifier()

        mock_hash = hashlib.sha256(b"test_block_1").digest()
        receipt = builder.build(
            block_hash=mock_hash,
            block_height=100,
            timestamp=1706700000,
            reward_amount=685410196,
            order_param=0.89,
            mean_phase=0.15,
            oscillator_count=63,
            nonce=42,
        )

        result = verifier.verify(receipt)

        assert result.valid
        assert result.payload is not None
        assert result.payload.block_height == 100
        assert result.payload.block_hash == mock_hash
        assert result.payload.reward_amount == 685410196
        assert result.archetype_name in ARCHETYPE_NAMES
        assert result.corridor_name in CORRIDOR_NAMES

    def test_verify_after_save_load(self):
        """Receipt survives PNG save/load cycle."""
        builder = ReceiptBuilder(size=63)
        verifier = ReceiptVerifier()

        mock_hash = hashlib.sha256(b"save_load_test").digest()
        original_height = 12345
        original_reward = 500000000

        receipt = builder.build(
            block_hash=mock_hash,
            block_height=original_height,
            timestamp=1706700000,
            reward_amount=original_reward,
            order_param=0.90,
            mean_phase=0.25,
            oscillator_count=63,
            nonce=999,
        )

        # Save to buffer
        buf = io.BytesIO()
        receipt.save(buf, format='PNG')
        buf.seek(0)

        # Reload
        loaded = Image.open(buf)

        # Verify
        result = verifier.verify(loaded)

        assert result.valid, f"Errors: {result.errors}"
        assert result.payload.block_height == original_height
        assert result.payload.block_hash == mock_hash
        assert result.payload.reward_amount == original_reward


class TestFullPipeline:
    """Full pipeline integration tests."""

    def test_full_pipeline(self):
        """Build and verify a receipt PNG."""
        builder = ReceiptBuilder(size=63)
        verifier = ReceiptVerifier()

        mock_hash = hashlib.sha256(b"full_pipeline_test").digest()
        receipt = builder.build(
            block_hash=mock_hash,
            block_height=1,
            timestamp=1706700000,
            reward_amount=685410196,
            order_param=0.89,
            mean_phase=0.15,
            oscillator_count=63,
            nonce=7,
        )

        # Save to buffer and reload (tests PNG lossless cycle)
        buf = io.BytesIO()
        receipt.save(buf, format="PNG")
        buf.seek(0)

        result = verifier.verify(Image.open(buf))

        assert result.valid
        assert result.payload.block_height == 1
        assert result.payload.block_hash == mock_hash
        assert result.payload.reward_amount == 685410196

    def test_different_blocks_different_images(self):
        """Different blocks produce different images."""
        builder = ReceiptBuilder(size=63)

        hash1 = hashlib.sha256(b"block_A").digest()
        hash2 = hashlib.sha256(b"block_B").digest()

        receipt1 = builder.build(
            block_hash=hash1,
            block_height=1,
            timestamp=1706700000,
            reward_amount=100000000,
            order_param=0.87,
            mean_phase=0.0,
            oscillator_count=63,
            nonce=0,
        )

        receipt2 = builder.build(
            block_hash=hash2,
            block_height=2,
            timestamp=1706700100,
            reward_amount=100000000,
            order_param=0.88,
            mean_phase=0.1,
            oscillator_count=63,
            nonce=1,
        )

        # Convert to arrays and compare
        arr1 = np.array(receipt1)
        arr2 = np.array(receipt2)

        # Should not be identical
        assert not np.array_equal(arr1, arr2)

    def test_visual_quality(self):
        """Center pixel is reasonably colored (not black/white)."""
        builder = ReceiptBuilder(size=63)

        mock_hash = hashlib.sha256(b"visual_test").digest()
        receipt = builder.build(
            block_hash=mock_hash,
            block_height=1,
            timestamp=1706700000,
            reward_amount=100000000,
            order_param=0.87,
            mean_phase=0.0,
            oscillator_count=63,
            nonce=0,
        )

        arr = np.array(receipt)
        center_pixel = arr[31, 31]  # Center of 63x63

        # Should not be pure black or pure white
        assert not np.all(center_pixel[:3] == 0), "Center is pure black"
        assert not np.all(center_pixel[:3] == 255), "Center is pure white"

        # Should have some color variation
        assert np.std(center_pixel[:3]) < 200, "Color channels too extreme"

    def test_tamper_detection(self):
        """Modifying pixels breaks verification."""
        builder = ReceiptBuilder(size=63)
        verifier = ReceiptVerifier()

        mock_hash = hashlib.sha256(b"tamper_test").digest()
        receipt = builder.build(
            block_hash=mock_hash,
            block_height=1,
            timestamp=1706700000,
            reward_amount=100000000,
            order_param=0.87,
            mean_phase=0.0,
            oscillator_count=63,
            nonce=0,
        )

        # Tamper with the image
        arr = np.array(receipt)
        # Flip alpha LSBs (where receipt data is stored)
        arr[0:10, 0:10, 3] ^= 0x03  # XOR both LSB bits

        tampered = Image.fromarray(arr, mode='RGBA')

        # Verification should fail
        result = verifier.verify(tampered)
        assert not result.valid

    def test_file_size_reasonable(self):
        """PNG file size is reasonable for 63x63 RGBA."""
        builder = ReceiptBuilder(size=63)

        mock_hash = hashlib.sha256(b"size_test").digest()
        receipt = builder.build(
            block_hash=mock_hash,
            block_height=1,
            timestamp=1706700000,
            reward_amount=100000000,
            order_param=0.87,
            mean_phase=0.0,
            oscillator_count=63,
            nonce=0,
        )

        buf = io.BytesIO()
        receipt.save(buf, format='PNG')
        size = buf.tell()

        # Should be less than 50KB for a 63x63 image
        assert size < 50000, f"File size {size} too large"

        # Should be at least 1KB (contains actual data)
        assert size > 1000, f"File size {size} too small"
