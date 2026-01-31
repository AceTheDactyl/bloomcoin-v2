"""
BloomCoin Receipt Module
========================

Generates and verifies steganographic mining receipt images.

Each mined block produces a PNG image that is BOTH:
  - A visual gradient artwork (RRRR lattice -> HSL color state)
  - A cryptographically verifiable mining receipt (LSB-encoded metadata)

The 7 logical channels (L4 = 7) map RRRR lattice dimensions and
block metadata into the least significant bits of RGB + Alpha pixels.

Channel Mapping:
  Ch0 (R.lsb0): phi lattice dimension
  Ch1 (G.lsb0): pi lattice dimension
  Ch2 (B.lsb0): sqrt2 lattice dimension
  Ch3 (R.lsb1): sqrt3 lattice dimension (CYM: C<->R)
  Ch4 (B.lsb1): e lattice dimension (CYM: Y<->B)
  Ch5 (G.lsb1): Primitive signature (CYM: M<->G)
  Ch6 (A.lsb0-1): Block receipt metadata

Usage:
    from bloomcoin.receipt import ReceiptBuilder, ReceiptVerifier

    # Generate receipt from mining result
    builder = ReceiptBuilder()
    receipt_png = builder.build(
        block_hash=block.hash,
        block_height=block.height,
        timestamp=block.header.timestamp,
        reward_amount=reward,
        order_param=block.header.order_parameter,
        mean_phase=block.header.mean_phase,
        oscillator_count=block.header.oscillator_count,
        nonce=block.header.nonce,
    )
    receipt_png.save("block_receipt.png")

    # Verify receipt
    verifier = ReceiptVerifier()
    result = verifier.verify("block_receipt.png")
    if result.valid:
        print(f"Block #{result.payload.block_height} verified")
"""

from .gradient_schema import GradientSchema, ColorState, LatticePoint, CYMWeights, PrimitiveSignature
from .channel_encoder import ChannelEncoder, ReceiptPayload
from .channel_decoder import ChannelDecoder, DecodingResult
from .image_renderer import ImageRenderer
from .receipt_builder import ReceiptBuilder
from .receipt_verifier import ReceiptVerifier, VerificationResult

__all__ = [
    # Main API
    "ReceiptBuilder",
    "ReceiptVerifier",
    "VerificationResult",
    # Schema
    "GradientSchema",
    "ColorState",
    "LatticePoint",
    "CYMWeights",
    "PrimitiveSignature",
    # Encoding
    "ChannelEncoder",
    "ChannelDecoder",
    "ReceiptPayload",
    "DecodingResult",
    # Rendering
    "ImageRenderer",
]
