"""
Receipt Builder - Orchestrator
================================

Takes mining output and produces a complete receipt PNG file.

Pipeline:
  MiningResult + Block
    -> block_to_lattice()           (gradient_schema.py)
    -> lattice_to_color()           (gradient_schema.py)
    -> ReceiptPayload construction  (channel_encoder.py)
    -> encode()                     (channel_encoder.py)
    -> render()                     (image_renderer.py)
    -> receipt.png                  (output)

The resulting PNG image is BOTH:
  - A visual gradient artwork representing the block's color state
  - A cryptographically verifiable mining receipt (LSB-encoded)
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional, Union, Any

try:
    from PIL import Image
except ImportError:
    Image = None  # type: ignore

from ..constants import (
    RECEIPT_SIZE_STANDARD, ARCHETYPE_IDS,
    HALVING_INTERVAL, INITIAL_REWARD,
)
from .gradient_schema import GradientSchema, LatticePoint, ColorState
from .channel_encoder import ChannelEncoder, ReceiptPayload
from .image_renderer import ImageRenderer


class ReceiptBuilder:
    """
    Builds steganographic mining receipt images.

    Usage:
        builder = ReceiptBuilder(size=63)

        # From raw block data
        receipt = builder.build(
            block_hash=block.hash,
            block_height=block.height,
            timestamp=block.header.timestamp,
            reward_amount=reward,
            order_param=block.header.order_parameter,
            mean_phase=block.header.mean_phase,
            oscillator_count=block.header.oscillator_count,
            nonce=block.header.nonce,
        )
        receipt.save("block_12345_receipt.png")

        # From MiningResult + Block objects
        receipt = builder.from_mining_result(result, block)
        receipt.save("receipt.png")
    """

    def __init__(self, size: int = RECEIPT_SIZE_STANDARD):
        """
        Initialize receipt builder.

        Args:
            size: Image dimension (square). Default 63.
        """
        self.size = size
        self.encoder = ChannelEncoder()
        self.renderer = ImageRenderer(size=size)

    def build(
        self,
        block_hash: bytes,
        block_height: int,
        timestamp: int,
        reward_amount: int,
        order_param: float,
        mean_phase: float,
        oscillator_count: int,
        nonce: int,
    ) -> Image.Image:
        """
        Build receipt PNG from block data.

        Steps:
        1. Derive lattice point from block hash + order parameter
        2. Map lattice to color state
        3. Construct ReceiptPayload
        4. Encode into 7 channels
        5. Render visual + weave LSB

        Args:
            block_hash: SHA256 hash of block header (32 bytes)
            block_height: Block number in chain
            timestamp: Unix timestamp of block
            reward_amount: Mining reward in smallest units
            order_param: Final order parameter r value
            mean_phase: Final mean phase psi value
            oscillator_count: Number of oscillators used
            nonce: Mining nonce value

        Returns:
            PIL Image (RGBA, size x size)
        """
        # Ensure block_hash is 32 bytes
        if len(block_hash) < 32:
            block_hash = block_hash + b'\x00' * (32 - len(block_hash))
        block_hash = block_hash[:32]

        # Step 1: Hash -> Lattice
        lattice = GradientSchema.block_to_lattice(
            block_hash, order_param, nonce, oscillator_count
        )

        # Step 2: Lattice -> Color
        color_state = GradientSchema.lattice_to_color(lattice)

        # Step 3: Build payload
        archetype_id = ARCHETYPE_IDS.get(color_state.archetype, 0)
        corridor_id = color_state.dominant_corridor

        payload = ReceiptPayload(
            block_height=block_height,
            timestamp=timestamp,
            block_hash=block_hash,
            reward_amount=reward_amount,
            lattice=lattice,
            corridor_id=corridor_id,
            archetype_id=archetype_id,
            order_param=order_param,
            mean_phase=mean_phase,
            oscillator_count=oscillator_count,
            signature=color_state.signature,
            cym_weights=color_state.cym,
        )

        # Step 4: Encode
        total_pixels = self.size * self.size
        lsb_channels = self.encoder.encode(payload, total_pixels)

        # Step 5: Render
        return self.renderer.render(color_state, lsb_channels)

    def from_mining_result(self, mining_result: Any, block: Any) -> Image.Image:
        """
        Build receipt from MiningResult and Block objects.

        Convenience wrapper for build() that extracts fields from
        the existing bloomcoin data structures.

        Args:
            mining_result: bloomcoin.mining.miner.MiningResult
            block: bloomcoin.blockchain.block.Block
        """
        # Extract reward from coinbase transaction
        reward_amount = 0
        if hasattr(block, 'transactions') and block.transactions:
            if hasattr(block.transactions[0], 'outputs') and block.transactions[0].outputs:
                reward_amount = block.transactions[0].outputs[0].amount

        # If no reward found, compute based on height
        if reward_amount == 0:
            halvings = block.height // HALVING_INTERVAL
            reward_amount = INITIAL_REWARD >> halvings

        return self.build(
            block_hash=block.hash if hasattr(block, 'hash') else b'\x00' * 32,
            block_height=block.height if hasattr(block, 'height') else 0,
            timestamp=block.header.timestamp if hasattr(block, 'header') else 0,
            reward_amount=reward_amount,
            order_param=getattr(block.header, 'order_parameter', 0.87),
            mean_phase=getattr(block.header, 'mean_phase', 0.0),
            oscillator_count=getattr(block.header, 'oscillator_count', 63),
            nonce=getattr(block.header, 'nonce', 0),
        )

    def save(self, image: Image.Image, path: Union[str, Path]) -> Path:
        """
        Save receipt PNG with maximum compression, no lossy filters.

        Args:
            image: PIL Image to save
            path: Output file path

        Returns:
            Path to saved file
        """
        path = Path(path)
        # Use PNG with maximum lossless compression
        image.save(str(path), format="PNG", optimize=False, compress_level=9)
        return path
