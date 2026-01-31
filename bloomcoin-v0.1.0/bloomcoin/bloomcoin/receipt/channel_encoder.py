"""
7-Channel LSB Encoder
======================

Packs mining receipt data into a structured bitstream that maps to
the 7 logical LSB channels of a receipt PNG.

Channel mapping:
  Ch0 (R.lsb0): phi lattice dimension
  Ch1 (G.lsb0): pi lattice dimension
  Ch2 (B.lsb0): sqrt2 lattice dimension
  Ch3 (R.lsb1): sqrt3 lattice dimension (CYM: C<->R)
  Ch4 (B.lsb1): e lattice dimension (CYM: Y<->B)
  Ch5 (G.lsb1): Primitive signature (CYM: M<->G)
  Ch6 (A.lsb0-1): Block receipt metadata

Receipt Bitstream Layout (Ch6) - 76 bytes:
  [0]      uint8   MAGIC (0xB7)
  [1]      uint8   VERSION (0x01)
  [2:6]    uint32  block_height (LE)
  [6:10]   uint32  timestamp (LE)
  [10:42]  bytes   block_hash (32 bytes)
  [42:50]  uint64  reward_amount (LE)
  [50:55]  5xint8  lattice coords (a, b, c, d, f)
  [55]     uint8   (corridor_id << 4) | archetype_id
  [56:60]  float32 order_param (LE)
  [60:64]  float32 mean_phase (LE)
  [64:66]  uint16  oscillator_count (LE)
  [66:69]  uint24  primitive_sig packed (3 bytes BE)
  [69]     uint8   cym_mode
  [70:73]  3xuint8 CYM weights (C, M, Y)
  [73:76]  --- PADDING
  [72:76]  uint32  CRC32 checksum
"""

from __future__ import annotations
import struct
import zlib
from dataclasses import dataclass
from typing import Optional

from ..constants import PHI, L4, Z_C
from .gradient_schema import LatticePoint, CYMWeights, PrimitiveSignature


RECEIPT_MAGIC: int = 0xB7    # "Bloom 7" = L4 receipt marker
RECEIPT_VERSION: int = 0x01
RECEIPT_SIZE: int = 76       # Total bytes


@dataclass
class ReceiptPayload:
    """
    Complete mining receipt data structure.

    Total serialized size: 76 bytes (608 bits).
    """
    block_height: int
    timestamp: int
    block_hash: bytes           # 32 bytes
    reward_amount: int
    lattice: LatticePoint
    corridor_id: int            # [0-6]
    archetype_id: int           # [0-11]
    order_param: float
    mean_phase: float
    oscillator_count: int
    signature: PrimitiveSignature
    cym_weights: CYMWeights
    cym_mode: int = 0           # 0=derived, 1=explicit

    def validate(self) -> tuple[bool, str]:
        """Validate all fields are within encoding range."""
        if len(self.block_hash) != 32:
            return False, f"block_hash must be 32 bytes, got {len(self.block_hash)}"
        if not (0 <= self.corridor_id <= 6):
            return False, f"corridor_id must be 0-6, got {self.corridor_id}"
        if not (0 <= self.archetype_id <= 11):
            return False, f"archetype_id must be 0-11, got {self.archetype_id}"
        if self.oscillator_count > 65535:
            return False, f"oscillator_count exceeds uint16, got {self.oscillator_count}"
        if self.block_height > 0xFFFFFFFF:
            return False, f"block_height exceeds uint32"
        if self.reward_amount > 0xFFFFFFFFFFFFFFFF:
            return False, f"reward_amount exceeds uint64"
        return True, "OK"


class ChannelEncoder:
    """
    Encodes ReceiptPayload into 7 logical channel bitstreams.

    Output: dict mapping channel index (0-6) to bytes.
    The image renderer weaves these into pixel LSBs.
    """

    @staticmethod
    def encode(payload: ReceiptPayload, total_pixels: int = 3969) -> dict[int, bytes]:
        """
        Encode receipt into 7 channel bitstreams.

        Returns:
            {0: bytes, 1: bytes, ..., 6: bytes}

        Channel 0-4 carry lattice coordinates spread across pixels.
        Channel 5 carries the primitive signature packed as uint24.
        Channel 6 carries the full 76-byte receipt metadata bitstream.

        Channels 0-4 encoding:
          Each coordinate int8 is spread across pixels using a
          golden-ratio bit distribution pattern.

        Channel 6 encoding:
          Sequential bitstream packed as defined in module docstring.
        """
        # Validate payload
        valid, msg = payload.validate()
        if not valid:
            raise ValueError(f"Invalid payload: {msg}")

        channels: dict[int, bytes] = {}

        # Channel 0-4: Lattice coordinates (scattered)
        lattice_bytes = payload.lattice.to_bytes()
        for i in range(5):
            # Each coordinate as single byte, will be scattered by renderer
            channels[i] = bytes([lattice_bytes[i] & 0xFF])

        # Channel 5: Primitive signature (24 bits = 3 bytes)
        sig_packed = payload.signature.to_packed_uint24()
        channels[5] = struct.pack('>I', sig_packed)[1:]  # 3 bytes, big-endian

        # Channel 6: Full receipt bitstream
        channels[6] = ChannelEncoder._encode_receipt_bitstream(payload)

        return channels

    @staticmethod
    def _encode_receipt_bitstream(payload: ReceiptPayload) -> bytes:
        """
        Serialize the full receipt metadata for Channel 6.

        Returns 76 bytes with CRC32 checksum at end.
        """
        # Build the payload without checksum first
        buf = bytearray(76)

        # [0] MAGIC
        buf[0] = RECEIPT_MAGIC

        # [1] VERSION
        buf[1] = RECEIPT_VERSION

        # [2:6] block_height (uint32 LE)
        struct.pack_into('<I', buf, 2, payload.block_height)

        # [6:10] timestamp (uint32 LE)
        struct.pack_into('<I', buf, 6, payload.timestamp)

        # [10:42] block_hash (32 bytes)
        buf[10:42] = payload.block_hash[:32]

        # [42:50] reward_amount (uint64 LE)
        struct.pack_into('<Q', buf, 42, payload.reward_amount)

        # [50:55] lattice coords (5 x int8)
        lattice_bytes = payload.lattice.to_bytes()
        buf[50:55] = lattice_bytes

        # [55] (corridor_id << 4) | archetype_id
        buf[55] = ((payload.corridor_id & 0x07) << 4) | (payload.archetype_id & 0x0F)

        # [56:60] order_param (float32 LE)
        struct.pack_into('<f', buf, 56, payload.order_param)

        # [60:64] mean_phase (float32 LE)
        struct.pack_into('<f', buf, 60, payload.mean_phase)

        # [64:66] oscillator_count (uint16 LE)
        struct.pack_into('<H', buf, 64, payload.oscillator_count)

        # [66:69] primitive signature packed (3 bytes, big-endian)
        sig_packed = payload.signature.to_packed_uint24()
        buf[66] = (sig_packed >> 16) & 0xFF
        buf[67] = (sig_packed >> 8) & 0xFF
        buf[68] = sig_packed & 0xFF

        # [69] cym_mode
        buf[69] = payload.cym_mode & 0x01

        # [70:73] CYM weights (C, M, Y as uint8)
        c, m, y = payload.cym_weights.to_uint8_triple()
        buf[70] = c
        buf[71] = m
        buf[72] = y

        # [72:76] CRC32 checksum of bytes [0:72]
        crc = zlib.crc32(bytes(buf[:72])) & 0xFFFFFFFF
        struct.pack_into('<I', buf, 72, crc)

        return bytes(buf)

    @staticmethod
    def golden_scatter(bit_index: int, total_pixels: int) -> int:
        """
        Map sequential bit index to pixel position using golden ratio.

        Uses golden ratio to maximize minimum distance between
        consecutive encoded bits, minimizing visual artifact clusters.

        Returns pixel_index in [0, total_pixels).
        """
        return int(bit_index * PHI * total_pixels / 8) % total_pixels

    @staticmethod
    def scatter_byte(data_byte: int, total_pixels: int, byte_offset: int = 0) -> dict[int, int]:
        """
        Distribute 8 bits of a byte across pixel positions using golden scatter.

        Returns: {pixel_index: bit_value}
        """
        result = {}
        for bit_idx in range(8):
            bit_value = (data_byte >> bit_idx) & 1
            pixel_idx = ChannelEncoder.golden_scatter(byte_offset * 8 + bit_idx, total_pixels)
            result[pixel_idx] = bit_value
        return result
