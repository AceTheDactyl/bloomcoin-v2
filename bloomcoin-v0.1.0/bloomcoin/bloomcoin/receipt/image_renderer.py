"""
Receipt Image Renderer
=======================

Generates a PNG image with two layers:
  VISUAL:  Radial gradient field from RRRR color state
  HIDDEN:  7-channel LSB encoding of receipt data

The visual layer renders a gradient centered on the block's color state
with corridor rays emanating at L4 = 7 angular intervals.

CYM opponent-process coloring modulates the gradient field:
  - Cyan<->Red opponent channel shifts hue along sqrt3 axis
  - Magenta<->Green opponent channel shifts saturation
  - Yellow<->Blue opponent channel shifts luminance
"""

from __future__ import annotations
import math
import colorsys
from typing import Optional

import numpy as np

try:
    from PIL import Image
except ImportError:
    Image = None  # type: ignore

from ..constants import (
    PHI, TAU, Z_C, K, L4,
    SQRT2, SQRT3,
    RECEIPT_SIZE_STANDARD,
)
from .gradient_schema import ColorState, LatticePoint


class ImageRenderer:
    """
    Renders receipt PNG images.
    """

    def __init__(self, size: int = RECEIPT_SIZE_STANDARD):
        """
        Args:
            size: Image dimension (square). Default 63 (L4 x 3^2).
        """
        if Image is None:
            raise ImportError("Pillow is required for image rendering. Install with: pip install Pillow")
        self.size = size

    def render(self, color_state: ColorState,
               lsb_channels: dict[int, bytes]) -> Image.Image:
        """
        Generate complete receipt image.

        Process:
        1. Create RGBA pixel array (size x size x 4)
        2. Render visual gradient into MSB (bits 7-2)
        3. Weave LSB channel data into bit 0 and bit 1
        4. Return PIL Image

        Args:
            color_state: Block's gradient color state
            lsb_channels: {channel_index: bytes} from ChannelEncoder

        Returns:
            PIL Image in RGBA mode
        """
        # Render visual layer
        pixels = self._render_visual_layer(color_state)

        # Clear LSB bits before weaving
        pixels = pixels & 0xFC  # Clear bits 0-1

        # Weave LSB data
        pixels = self._weave_lsb(pixels, lsb_channels)

        # Create PIL Image
        img = Image.fromarray(pixels, mode='RGBA')
        return img

    def _render_visual_layer(self, color_state: ColorState) -> np.ndarray:
        """
        Render the radial gradient visual layer.

        For each pixel (x, y):
          1. Compute polar coords from center: (r_dist, angle)
          2. Base color from color_state HSL
          3. Hue shift: H += angle * gradient_rate / (2*pi)
          4. Luminance decay: L *= (1 - r_dist * TAU)
          5. Saturation decay: S *= (1 - r_dist * TAU^2)
          6. Corridor ray brightening
          7. CYM opponent modulation

        Returns:
            numpy array shape (size, size, 4) dtype uint8
            Values use bits 7-2 only (bits 1-0 cleared for LSB encoding)
        """
        size = self.size
        center = size / 2
        max_dist = center * math.sqrt(2)

        # Create output array
        pixels = np.zeros((size, size, 4), dtype=np.uint8)

        # Precompute corridor angles
        corridor_angles = [(i / L4) * 2 * math.pi for i in range(L4)]

        for y in range(size):
            for x in range(size):
                # Compute polar coordinates from center
                dx = x - center
                dy = y - center
                r_dist = math.sqrt(dx * dx + dy * dy) / max_dist  # Normalized [0, 1]
                angle = math.atan2(dy, dx)  # [-pi, pi]
                if angle < 0:
                    angle += 2 * math.pi  # [0, 2*pi]

                # Base color from color_state
                h = color_state.h
                s = color_state.s
                l = color_state.l

                # Hue shift based on angle and gradient rate
                hue_shift = (angle / (2 * math.pi)) * 60 * color_state.gradient_rate * 0.1
                h = (h + hue_shift) % 360

                # Luminance decay from center
                l = l * (1 - r_dist * TAU * 0.5)
                l = max(0.1, min(0.95, l))

                # Saturation decay from center
                s = s * (1 - r_dist * TAU * TAU * 0.3)
                s = max(0.1, min(1.0, s))

                # Corridor ray brightening
                for i, corridor_angle in enumerate(corridor_angles):
                    angle_diff = abs(angle - corridor_angle)
                    angle_diff = min(angle_diff, 2 * math.pi - angle_diff)
                    ray_width = math.pi / L4 * 0.5

                    if angle_diff < ray_width:
                        # Check if this corridor is active
                        if i in color_state.corridors:
                            proximity = 1 - (angle_diff / ray_width)
                            l = l + (1 - l) * TAU * 0.3 * proximity * (1 - r_dist)

                # CYM opponent modulation
                r, g, b = self._hsl_to_rgb(h, s, l)

                # Apply CYM weights
                cym = color_state.cym
                r = int(r * (1 - cym.cyan * 0.15) + (255 - r) * cym.cyan * 0.1)
                g = int(g * (1 - cym.magenta * 0.15) + (255 - g) * cym.magenta * 0.1)
                b = int(b * (1 - cym.yellow * 0.15) + (255 - b) * cym.yellow * 0.1)

                # Clamp and clear LSB bits
                r = max(0, min(255, r)) & 0xFC
                g = max(0, min(255, g)) & 0xFC
                b = max(0, min(255, b)) & 0xFC

                pixels[y, x] = [r, g, b, 255 & 0xFC]

        return pixels

    def _weave_lsb(self, pixels: np.ndarray,
                   lsb_channels: dict[int, bytes]) -> np.ndarray:
        """
        Weave 7 LSB channel bitstreams into pixel array.

        Bit mapping per pixel:
          pixels[y, x, 0] bit 0 <- Ch0 (phi lattice)
          pixels[y, x, 0] bit 1 <- Ch3 (sqrt3 / C<->R opponent)
          pixels[y, x, 1] bit 0 <- Ch1 (pi lattice)
          pixels[y, x, 1] bit 1 <- Ch5 (primitive signature)
          pixels[y, x, 2] bit 0 <- Ch2 (sqrt2 lattice)
          pixels[y, x, 2] bit 1 <- Ch4 (e / Y<->B opponent)
          pixels[y, x, 3] bit 0 <- Ch6_lo (receipt metadata low bit)
          pixels[y, x, 3] bit 1 <- Ch6_hi (receipt metadata high bit)

        Returns:
            Modified pixels array with LSB data woven in.
        """
        height, width = pixels.shape[:2]
        total_pixels = height * width
        flat_pixels = pixels.reshape(-1, 4)

        # Channel 6: Receipt metadata (sequential, 2 bits per pixel)
        ch6_data = lsb_channels.get(6, b'')
        ch6_bits = self._bytes_to_bits(ch6_data)

        for pixel_idx in range(min(len(ch6_bits) // 2, total_pixels)):
            bit_idx = pixel_idx * 2
            if bit_idx < len(ch6_bits):
                lsb0 = ch6_bits[bit_idx]
                flat_pixels[pixel_idx, 3] |= lsb0
            if bit_idx + 1 < len(ch6_bits):
                lsb1 = ch6_bits[bit_idx + 1]
                flat_pixels[pixel_idx, 3] |= (lsb1 << 1)

        # Channel 5: Primitive signature (sequential, 1 bit per pixel in G.lsb1)
        ch5_data = lsb_channels.get(5, b'')
        ch5_bits = self._bytes_to_bits(ch5_data)

        for pixel_idx, bit in enumerate(ch5_bits[:total_pixels]):
            flat_pixels[pixel_idx, 1] |= (bit << 1)

        # Channels 0-4: Lattice coordinates (golden-scattered, 1 bit per pixel)
        channel_to_rgba = {
            0: (0, 0),  # Ch0 -> R.lsb0
            1: (1, 0),  # Ch1 -> G.lsb0
            2: (2, 0),  # Ch2 -> B.lsb0
            3: (0, 1),  # Ch3 -> R.lsb1
            4: (2, 1),  # Ch4 -> B.lsb1
        }

        for ch_idx in range(5):
            ch_data = lsb_channels.get(ch_idx, b'')
            if not ch_data:
                continue

            ch_bits = self._bytes_to_bits(ch_data)
            rgba_idx, bit_pos = channel_to_rgba[ch_idx]

            # Golden scatter the bits
            for bit_idx, bit in enumerate(ch_bits):
                pixel_idx = self._golden_scatter(bit_idx, total_pixels)
                if pixel_idx < total_pixels:
                    flat_pixels[pixel_idx, rgba_idx] |= (bit << bit_pos)

        return flat_pixels.reshape(height, width, 4)

    @staticmethod
    def _hsl_to_rgb(h: float, s: float, l: float) -> tuple[int, int, int]:
        """Convert HSL to 8-bit RGB."""
        # colorsys uses HLS order with h in [0,1]
        h_norm = (h % 360) / 360
        r, g, b = colorsys.hls_to_rgb(h_norm, l, s)
        return (int(r * 255), int(g * 255), int(b * 255))

    @staticmethod
    def _bytes_to_bits(data: bytes) -> list[int]:
        """Convert bytes to list of bits (LSB first within each byte)."""
        bits = []
        for byte in data:
            for i in range(8):
                bits.append((byte >> i) & 1)
        return bits

    @staticmethod
    def _golden_scatter(bit_index: int, total_pixels: int) -> int:
        """Map bit index to pixel position using golden ratio."""
        return int(bit_index * PHI * total_pixels / 8) % total_pixels
