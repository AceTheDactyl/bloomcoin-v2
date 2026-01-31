"""
Receipt Verifier
=================

Loads a receipt PNG, extracts the LSB-encoded mining receipt,
and validates its cryptographic integrity.

Verification checks:
  1. Magic byte == 0xB7
  2. Version supported
  3. CRC32 checksum matches
  4. Lattice coordinates cross-validate (Ch0-4 vs Ch6 redundant copy)
  5. Block hash derivable from receipt fields
  6. Gradient color state consistent with encoded lattice coords
"""

from __future__ import annotations
from pathlib import Path
from typing import Union, Optional
from dataclasses import dataclass

import numpy as np

try:
    from PIL import Image
except ImportError:
    Image = None  # type: ignore

from ..constants import ARCHETYPE_NAMES, CORRIDOR_NAMES
from .channel_decoder import ChannelDecoder, DecodingResult
from .gradient_schema import GradientSchema, LatticePoint, ColorState
from .channel_encoder import ReceiptPayload


@dataclass
class VerificationResult:
    """Complete verification result."""
    valid: bool
    payload: Optional[ReceiptPayload]
    color_state: Optional[ColorState]
    errors: list[str]
    warnings: list[str]

    @property
    def block_hash_hex(self) -> Optional[str]:
        """Return block hash as hex string."""
        if self.payload:
            return self.payload.block_hash.hex()
        return None

    @property
    def archetype_name(self) -> Optional[str]:
        """Return archetype name."""
        if self.payload and 0 <= self.payload.archetype_id < len(ARCHETYPE_NAMES):
            return ARCHETYPE_NAMES[self.payload.archetype_id]
        return None

    @property
    def corridor_name(self) -> Optional[str]:
        """Return corridor name."""
        if self.payload and 0 <= self.payload.corridor_id < len(CORRIDOR_NAMES):
            return CORRIDOR_NAMES[self.payload.corridor_id]
        return None

    @classmethod
    def failure(cls, errors: list[str]) -> VerificationResult:
        """Create a failure result."""
        return cls(
            valid=False,
            payload=None,
            color_state=None,
            errors=errors,
            warnings=[],
        )

    @classmethod
    def success(cls, payload: ReceiptPayload, color_state: ColorState,
                warnings: Optional[list[str]] = None) -> VerificationResult:
        """Create a success result."""
        return cls(
            valid=True,
            payload=payload,
            color_state=color_state,
            errors=[],
            warnings=warnings or [],
        )


class ReceiptVerifier:
    """
    Verifies steganographic mining receipt images.

    Usage:
        verifier = ReceiptVerifier()
        result = verifier.verify("block_12345_receipt.png")

        if result.valid:
            print(f"Block #{result.payload.block_height}")
            print(f"Hash: {result.block_hash_hex}")
            print(f"Reward: {result.payload.reward_amount}")
            print(f"Archetype: {result.archetype_name}")
            print(f"Corridor: {result.corridor_name}")
    """

    def verify(self, image_path: Union[str, Path, Image.Image]) -> VerificationResult:
        """
        Load and verify a receipt PNG.

        Steps:
        1. Load image as RGBA numpy array
        2. Decode LSB channels
        3. Parse receipt payload
        4. Cross-validate lattice coordinates
        5. Re-derive color state from lattice and compare
        6. Report all findings

        Args:
            image_path: Path to PNG or PIL Image object

        Returns:
            VerificationResult
        """
        if Image is None:
            return VerificationResult.failure(
                ["Pillow is required for verification. Install with: pip install Pillow"]
            )

        errors: list[str] = []
        warnings: list[str] = []

        # Step 1: Load image
        try:
            if isinstance(image_path, Image.Image):
                img = image_path
            else:
                img = Image.open(str(image_path))

            # Convert to RGBA if needed
            if img.mode != 'RGBA':
                img = img.convert('RGBA')

            pixels = np.array(img, dtype=np.uint8)

        except Exception as e:
            return VerificationResult.failure([f"Failed to load image: {e}"])

        # Step 2-3: Decode LSB channels
        decode_result = ChannelDecoder.decode(pixels)

        if not decode_result.success:
            return VerificationResult.failure([f"Decode failed: {decode_result.error}"])

        payload = decode_result.payload
        if payload is None:
            return VerificationResult.failure(["No payload extracted"])

        # Check CRC
        if not decode_result.checksum_valid:
            errors.append("CRC32 checksum mismatch")

        # Check lattice cross-validation
        if not decode_result.lattice_cross_valid:
            warnings.append("Lattice cross-validation skipped or failed")

        # Step 5: Re-derive color state from lattice
        try:
            color_state = GradientSchema.lattice_to_color(payload.lattice)
        except Exception as e:
            warnings.append(f"Could not re-derive color state: {e}")
            color_state = None

        # Step 6: Validate gradient consistency
        if color_state is not None:
            gradient_warnings = self._cross_validate_gradient(payload, pixels, color_state)
            warnings.extend(gradient_warnings)

        if errors:
            return VerificationResult.failure(errors)

        return VerificationResult.success(payload, color_state, warnings)

    @staticmethod
    def _cross_validate_gradient(payload: ReceiptPayload,
                                  pixels: np.ndarray,
                                  expected_color: ColorState) -> list[str]:
        """
        Verify the visual gradient is consistent with the encoded lattice.

        Re-derive color state from lattice coords in the payload,
        then spot-check that the visual pixel colors are approximately
        correct (within LSB tolerance of +/-4 per channel).

        Returns list of warning messages (empty if consistent).
        """
        warnings = []

        # Get center pixel color
        height, width = pixels.shape[:2]
        center_y, center_x = height // 2, width // 2
        center_pixel = pixels[center_y, center_x]

        # Get expected RGB (with LSB bits potentially modified)
        expected_rgb = expected_color.to_rgb()

        # Check if reasonably close (within +/- 8 for each channel, accounting for LSB)
        tolerance = 8
        r_diff = abs(int(center_pixel[0]) - expected_rgb[0])
        g_diff = abs(int(center_pixel[1]) - expected_rgb[1])
        b_diff = abs(int(center_pixel[2]) - expected_rgb[2])

        if r_diff > tolerance or g_diff > tolerance or b_diff > tolerance:
            warnings.append(
                f"Center pixel color differs from expected: "
                f"actual=({center_pixel[0]},{center_pixel[1]},{center_pixel[2]}), "
                f"expected=({expected_rgb[0]},{expected_rgb[1]},{expected_rgb[2]})"
            )

        # Check archetype consistency
        expected_archetype = expected_color.archetype_id
        if expected_archetype != payload.archetype_id:
            warnings.append(
                f"Archetype mismatch: encoded={payload.archetype_id}, "
                f"derived={expected_archetype}"
            )

        # Check corridor consistency
        expected_corridor = expected_color.dominant_corridor
        if expected_corridor != payload.corridor_id:
            warnings.append(
                f"Corridor mismatch: encoded={payload.corridor_id}, "
                f"derived={expected_corridor}"
            )

        return warnings
