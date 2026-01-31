"""
Multi-Resolution Phase (MRP) Encoding with LSB Steganography

This module bridges continuous phase dynamics to discrete RGB output using:
1. Multi-Resolution Phase (MRP) quantization for gradient encoding
2. LSB steganography for embedding metadata and payloads
3. Hexagonal wavevector encoding for spatial navigation

The MRP scheme splits phase values into coarse (spatial bin) and fine
(sub-bin angle) components, enabling both robust low-resolution recovery
and high-precision encoding when all bits are available.
"""

import math
import struct
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

# =============================================================================
# Constants
# =============================================================================

TWO_PI = 2 * math.pi

# 8-bit MRP configuration (standard)
COARSE_BITS = 5    # Upper 5 bits: 32 phase bins
FINE_BITS = 3      # Lower 3 bits: 8 sub-bins

# 16-bit MRP configuration (high precision)
COARSE_BITS_16 = 10
FINE_BITS_16 = 6

# CRC-8 polynomial (standard CCITT)
CRC8_POLY = 0x07

# Magic bytes for MRP frame identification
MRP_MAGIC = b'\x4C\x34'  # "L4"


# =============================================================================
# Phase Quantization Functions
# =============================================================================

def quantize_phase(phase: float, bits: int = 8, wrap: bool = True) -> int:
    """
    Quantize a continuous phase value [0, 2*pi) to an integer.

    Args:
        phase: Phase value in radians
        bits: Number of bits for quantization (determines resolution)
        wrap: If True, wrap phase to [0, 2*pi) before quantizing

    Returns:
        Quantized integer in range [0, 2^bits - 1]
    """
    if wrap:
        phase = phase % TWO_PI

    max_val = (1 << bits) - 1
    # Map [0, 2*pi) -> [0, max_val]
    quantized = int((phase / TWO_PI) * (max_val + 1))
    return min(quantized, max_val)


def dequantize_phase(value: int, bits: int = 8) -> float:
    """
    Convert a quantized integer back to a continuous phase value.

    Args:
        value: Quantized integer
        bits: Number of bits used in quantization

    Returns:
        Phase value in radians [0, 2*pi)
    """
    max_val = (1 << bits)
    # Map [0, 2^bits - 1] -> [0, 2*pi)
    # Add 0.5 to center in bin
    return ((value + 0.5) / max_val) * TWO_PI


def quantize_phases_rgb(
    phases: Tuple[float, float, float],
    bits: int = 8
) -> Tuple[int, int, int]:
    """
    Quantize three phase values to RGB integer values.

    Args:
        phases: Tuple of (phase_r, phase_g, phase_b) in radians
        bits: Number of bits per channel

    Returns:
        Tuple of (R, G, B) quantized values
    """
    return tuple(quantize_phase(p, bits) for p in phases)


def dequantize_rgb_phases(
    rgb: Tuple[int, int, int],
    bits: int = 8
) -> Tuple[float, float, float]:
    """
    Convert RGB integer values back to phase values.

    Args:
        rgb: Tuple of (R, G, B) quantized values
        bits: Number of bits per channel

    Returns:
        Tuple of (phase_r, phase_g, phase_b) in radians
    """
    return tuple(dequantize_phase(v, bits) for v in rgb)


# =============================================================================
# MRPValue Dataclass
# =============================================================================

@dataclass
class MRPValue:
    """
    Multi-Resolution Phase value with coarse and fine components.

    The coarse bits represent the major spatial bin, while fine bits
    provide sub-bin angular precision. This allows graceful degradation
    under lossy conditions while maintaining high precision when all
    bits are available.
    """
    coarse: int       # Upper bits (spatial bin index)
    fine: int         # Lower bits (sub-bin angle)
    full: int         # Combined value (coarse << fine_bits | fine)
    phase: float      # Original continuous phase value

    @property
    def resolution_deg(self) -> Tuple[float, float]:
        """
        Get the angular resolution in degrees for coarse and fine components.

        Returns:
            Tuple of (coarse_resolution_deg, fine_resolution_deg)
        """
        # Calculate bits from full value
        total_bits = self.full.bit_length() if self.full > 0 else 8
        if total_bits <= 8:
            coarse_bits = COARSE_BITS
            fine_bits = FINE_BITS
        else:
            coarse_bits = COARSE_BITS_16
            fine_bits = FINE_BITS_16

        coarse_bins = 1 << coarse_bits
        fine_bins = 1 << fine_bits

        coarse_res = 360.0 / coarse_bins
        fine_res = coarse_res / fine_bins

        return (coarse_res, fine_res)


# =============================================================================
# MRP Encoding/Decoding
# =============================================================================

def encode_mrp(
    phase: float,
    coarse_bits: int = COARSE_BITS,
    fine_bits: int = FINE_BITS
) -> MRPValue:
    """
    Encode a phase value using Multi-Resolution Phase encoding.

    Args:
        phase: Phase value in radians [0, 2*pi)
        coarse_bits: Number of bits for coarse (bin) component
        fine_bits: Number of bits for fine (sub-bin) component

    Returns:
        MRPValue with encoded components
    """
    # Normalize phase to [0, 2*pi)
    phase = phase % TWO_PI

    total_bits = coarse_bits + fine_bits
    coarse_bins = 1 << coarse_bits
    fine_bins = 1 << fine_bits
    total_bins = 1 << total_bits

    # Quantize to full precision
    full = int((phase / TWO_PI) * total_bins)
    full = min(full, total_bins - 1)

    # Extract coarse and fine components
    coarse = full >> fine_bits
    fine = full & ((1 << fine_bits) - 1)

    return MRPValue(
        coarse=coarse,
        fine=fine,
        full=full,
        phase=phase
    )


def decode_mrp(
    mrp: MRPValue,
    coarse_bits: int = COARSE_BITS,
    fine_bits: int = FINE_BITS
) -> float:
    """
    Decode an MRPValue back to a continuous phase value.

    Args:
        mrp: MRPValue to decode
        coarse_bits: Number of bits for coarse component
        fine_bits: Number of bits for fine component

    Returns:
        Phase value in radians [0, 2*pi)
    """
    total_bits = coarse_bits + fine_bits
    total_bins = 1 << total_bits

    # Center in bin by adding 0.5
    return ((mrp.full + 0.5) / total_bins) * TWO_PI


def decode_coarse_only(mrp: MRPValue, coarse_bits: int = COARSE_BITS) -> float:
    """
    Decode using only the coarse component (lossy but robust).

    This is useful when fine bits may be corrupted or used for
    steganographic payload.

    Args:
        mrp: MRPValue to decode
        coarse_bits: Number of coarse bits

    Returns:
        Phase value in radians (lower precision)
    """
    coarse_bins = 1 << coarse_bits
    # Center in coarse bin
    return ((mrp.coarse + 0.5) / coarse_bins) * TWO_PI


# =============================================================================
# LSB Steganography Functions
# =============================================================================

def get_lsb(value: int, n_bits: int = 1) -> int:
    """
    Extract the least significant bits from a value.

    Args:
        value: Integer value to extract from
        n_bits: Number of LSBs to extract

    Returns:
        The extracted LSB value
    """
    mask = (1 << n_bits) - 1
    return value & mask


def set_lsb(value: int, payload: int, n_bits: int = 1) -> int:
    """
    Set the least significant bits of a value.

    Args:
        value: Original integer value
        payload: Payload to embed in LSBs
        n_bits: Number of LSBs to set

    Returns:
        Modified value with payload in LSBs
    """
    mask = (1 << n_bits) - 1
    payload = payload & mask  # Ensure payload fits
    # Clear LSBs, then set them
    return (value & ~mask) | payload


def embed_lsb_rgb(
    rgb: Tuple[int, int, int],
    payload: Tuple[int, int, int],
    n_bits: int = 3
) -> Tuple[int, int, int]:
    """
    Embed payload bits into RGB LSBs.

    Args:
        rgb: Original (R, G, B) tuple
        payload: Tuple of (payload_r, payload_g, payload_b) to embed
        n_bits: Number of LSBs to use per channel

    Returns:
        Modified RGB tuple with embedded payload
    """
    return tuple(
        set_lsb(c, p, n_bits)
        for c, p in zip(rgb, payload)
    )


def extract_lsb_rgb(
    rgb: Tuple[int, int, int],
    n_bits: int = 3
) -> Tuple[int, int, int]:
    """
    Extract LSB payload from RGB values.

    Args:
        rgb: (R, G, B) tuple with embedded payload
        n_bits: Number of LSBs per channel

    Returns:
        Tuple of extracted (payload_r, payload_g, payload_b)
    """
    return tuple(get_lsb(c, n_bits) for c in rgb)


# =============================================================================
# CRC-8 and Parity Functions
# =============================================================================

def crc8(data: bytes) -> int:
    """
    Compute CRC-8 checksum using standard polynomial.

    Args:
        data: Bytes to compute checksum for

    Returns:
        8-bit CRC value
    """
    crc = 0x00
    for byte in data:
        crc ^= byte
        for _ in range(8):
            if crc & 0x80:
                crc = ((crc << 1) ^ CRC8_POLY) & 0xFF
            else:
                crc = (crc << 1) & 0xFF
    return crc


def xor_parity(values: List[int]) -> int:
    """
    Compute XOR parity of a list of values.

    Args:
        values: List of integer values

    Returns:
        XOR of all values
    """
    result = 0
    for v in values:
        result ^= v
    return result


def compute_rgb_parity(r: int, g: int, n_bits: int = 3) -> int:
    """
    Compute parity value for RGB encoding.

    The parity is computed from the fine bits of R and G channels,
    and stored in the B channel's LSBs.

    Args:
        r: Red channel value
        g: Green channel value
        n_bits: Number of LSBs to use

    Returns:
        Parity value to store in B channel
    """
    r_fine = get_lsb(r, n_bits)
    g_fine = get_lsb(g, n_bits)
    return r_fine ^ g_fine


def verify_rgb_parity(rgb: Tuple[int, int, int], n_bits: int = 3) -> bool:
    """
    Verify RGB parity is consistent.

    Args:
        rgb: (R, G, B) tuple to verify
        n_bits: Number of LSBs used for parity

    Returns:
        True if parity is valid, False otherwise
    """
    r, g, b = rgb
    expected_parity = compute_rgb_parity(r, g, n_bits)
    actual_parity = get_lsb(b, n_bits)
    return expected_parity == actual_parity


# =============================================================================
# MRPFrame Dataclass and Encoding
# =============================================================================

@dataclass
class MRPFrame:
    """
    A complete MRP-encoded frame with RGB output and metadata.

    Contains the original phases, resulting RGB values, MRP components,
    parity verification status, and CRC checksum.
    """
    phases: Tuple[float, float, float]
    rgb: Tuple[int, int, int]
    mrp: Tuple[MRPValue, MRPValue, MRPValue]
    parity_ok: bool
    crc: int

    def to_bytes(self) -> bytes:
        """
        Serialize frame to bytes for storage/transmission.

        Format:
        - 2 bytes: Magic (MRP_MAGIC)
        - 3 bytes: RGB values
        - 12 bytes: 3 phases as float32 (little-endian)
        - 1 byte: Flags (parity_ok in bit 0)
        - 1 byte: CRC

        Returns:
            Serialized frame (19 bytes)
        """
        data = bytearray()

        # Magic bytes
        data.extend(MRP_MAGIC)

        # RGB values
        data.extend(bytes(self.rgb))

        # Phases as float32
        for phase in self.phases:
            data.extend(struct.pack('<f', phase))

        # Flags
        flags = 0x01 if self.parity_ok else 0x00
        data.append(flags)

        # CRC (computed over everything except CRC itself)
        computed_crc = crc8(bytes(data))
        data.append(computed_crc)

        return bytes(data)

    @classmethod
    def from_bytes(
        cls,
        data: bytes,
        phases: Optional[Tuple[float, float, float]] = None
    ) -> 'MRPFrame':
        """
        Deserialize frame from bytes.

        Args:
            data: Serialized frame bytes
            phases: Optional original phases (if not stored in data)

        Returns:
            Reconstructed MRPFrame

        Raises:
            ValueError: If magic bytes don't match or CRC fails
        """
        if len(data) < 19:
            raise ValueError(f"Data too short: {len(data)} bytes, need 19")

        # Verify magic
        if data[:2] != MRP_MAGIC:
            raise ValueError(f"Invalid magic: {data[:2].hex()}")

        # Extract RGB
        rgb = tuple(data[2:5])

        # Extract phases
        stored_phases = tuple(
            struct.unpack('<f', data[5 + i*4:9 + i*4])[0]
            for i in range(3)
        )

        # Use provided phases or stored phases
        if phases is not None:
            final_phases = phases
        else:
            final_phases = stored_phases

        # Extract flags
        flags = data[17]
        parity_ok = bool(flags & 0x01)

        # Verify CRC
        stored_crc = data[18]
        computed_crc = crc8(data[:18])

        if stored_crc != computed_crc:
            raise ValueError(
                f"CRC mismatch: stored={stored_crc:02x}, "
                f"computed={computed_crc:02x}"
            )

        # Reconstruct MRP values
        mrp = tuple(
            encode_mrp(p) for p in final_phases
        )

        return cls(
            phases=final_phases,
            rgb=rgb,
            mrp=mrp,
            parity_ok=parity_ok,
            crc=stored_crc
        )


def encode_frame(
    phases: Tuple[float, float, float],
    embed_parity: bool = True
) -> MRPFrame:
    """
    Encode three phase values into an MRP frame.

    Args:
        phases: Tuple of (phase_r, phase_g, phase_b) in radians
        embed_parity: If True, embed parity in B channel LSBs

    Returns:
        Complete MRPFrame with RGB output
    """
    # Encode each phase using MRP
    mrp_values = tuple(encode_mrp(p) for p in phases)

    # Get full quantized values as RGB
    rgb = [mrp.full for mrp in mrp_values]

    # Optionally embed parity in B channel
    if embed_parity:
        parity = compute_rgb_parity(rgb[0], rgb[1], FINE_BITS)
        rgb[2] = set_lsb(rgb[2], parity, FINE_BITS)

    rgb = tuple(rgb)

    # Verify parity
    parity_ok = verify_rgb_parity(rgb, FINE_BITS)

    # Compute CRC over phases
    phase_bytes = b''.join(struct.pack('<f', p) for p in phases)
    frame_crc = crc8(phase_bytes)

    return MRPFrame(
        phases=phases,
        rgb=rgb,
        mrp=mrp_values,
        parity_ok=parity_ok,
        crc=frame_crc
    )


def decode_frame(
    rgb: Tuple[int, int, int],
    verify: bool = True
) -> Tuple[Tuple[float, float, float], bool]:
    """
    Decode RGB values back to phases.

    Args:
        rgb: (R, G, B) tuple to decode
        verify: If True, verify parity

    Returns:
        Tuple of (phases, parity_ok) where phases is (r, g, b) in radians
    """
    # Verify parity if requested
    parity_ok = True
    if verify:
        parity_ok = verify_rgb_parity(rgb, FINE_BITS)

    # Decode each channel
    mrp_values = [
        MRPValue(
            coarse=v >> FINE_BITS,
            fine=v & ((1 << FINE_BITS) - 1),
            full=v,
            phase=0.0  # Will be computed
        )
        for v in rgb
    ]

    # Convert to phases
    phases = tuple(decode_mrp(mrp) for mrp in mrp_values)

    return phases, parity_ok


# =============================================================================
# Image-Level Encoding/Decoding
# =============================================================================

def encode_phase_image(
    phases_array: np.ndarray,
    embed_parity: bool = True
) -> np.ndarray:
    """
    Encode a 3D array of phases into an RGB image.

    Args:
        phases_array: Shape (H, W, 3) array of phases in radians
        embed_parity: If True, embed parity in B channel

    Returns:
        Shape (H, W, 3) uint8 RGB image
    """
    if phases_array.ndim != 3 or phases_array.shape[2] != 3:
        raise ValueError(
            f"Expected shape (H, W, 3), got {phases_array.shape}"
        )

    H, W, _ = phases_array.shape
    image = np.zeros((H, W, 3), dtype=np.uint8)

    for y in range(H):
        for x in range(W):
            phases = tuple(phases_array[y, x])
            frame = encode_frame(phases, embed_parity)
            image[y, x] = frame.rgb

    return image


def decode_phase_image(
    image: np.ndarray,
    verify: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Decode an RGB image back to phases.

    Args:
        image: Shape (H, W, 3) uint8 RGB image
        verify: If True, verify parity for each pixel

    Returns:
        Tuple of (phases_array, parity_mask) where:
        - phases_array: Shape (H, W, 3) float32 phases in radians
        - parity_mask: Shape (H, W) boolean array (True where parity OK)
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected shape (H, W, 3), got {image.shape}")

    H, W, _ = image.shape
    phases = np.zeros((H, W, 3), dtype=np.float32)
    parity_mask = np.zeros((H, W), dtype=bool)

    for y in range(H):
        for x in range(W):
            rgb = tuple(image[y, x])
            decoded_phases, parity_ok = decode_frame(rgb, verify)
            phases[y, x] = decoded_phases
            parity_mask[y, x] = parity_ok

    return phases, parity_mask


# =============================================================================
# Payload Steganography
# =============================================================================

def embed_payload_in_image(
    image: np.ndarray,
    payload: bytes,
    bits_per_channel: int = 2
) -> np.ndarray:
    """
    Embed a byte payload into an image using LSB steganography.

    Format:
    - First 4 bytes: Magic (MRP_MAGIC + length high byte + length low byte)
    - Next 2 bytes: Payload length (big-endian)
    - Remaining: Payload data
    - Final byte: CRC-8

    Args:
        image: Source RGB image (will be copied)
        payload: Bytes to embed
        bits_per_channel: LSBs to use per channel (1-4)

    Returns:
        Image with embedded payload

    Raises:
        ValueError: If payload too large for image capacity
    """
    if bits_per_channel < 1 or bits_per_channel > 4:
        raise ValueError("bits_per_channel must be 1-4")

    # Calculate capacity
    H, W, C = image.shape
    bits_per_pixel = bits_per_channel * C
    total_bits = H * W * bits_per_pixel
    total_bytes = total_bits // 8

    # Build complete message
    message = bytearray()
    message.extend(MRP_MAGIC)
    message.extend(struct.pack('>H', len(payload)))  # Big-endian length
    message.extend(payload)
    message.append(crc8(bytes(message)))  # CRC of everything before

    if len(message) > total_bytes:
        raise ValueError(
            f"Payload too large: {len(message)} bytes, "
            f"capacity {total_bytes} bytes"
        )

    # Convert message to bits
    bits = []
    for byte in message:
        for i in range(7, -1, -1):
            bits.append((byte >> i) & 1)

    # Pad with zeros
    while len(bits) < total_bits:
        bits.append(0)

    # Embed bits into image
    result = image.copy()
    bit_idx = 0
    mask = (1 << bits_per_channel) - 1
    inv_mask = 255 - mask  # Use subtraction instead of ~ for uint8 compatibility

    for y in range(H):
        for x in range(W):
            for c in range(C):
                if bit_idx + bits_per_channel <= len(bits):
                    # Extract bits_per_channel bits
                    value = 0
                    for b in range(bits_per_channel):
                        if bit_idx < len(bits):
                            value = (value << 1) | bits[bit_idx]
                            bit_idx += 1

                    # Embed in LSBs
                    result[y, x, c] = (int(result[y, x, c]) & inv_mask) | value

    return result


def extract_payload_from_image(
    image: np.ndarray,
    bits_per_channel: int = 2
) -> Optional[bytes]:
    """
    Extract a byte payload from an image.

    Args:
        image: Image with embedded payload
        bits_per_channel: LSBs used per channel

    Returns:
        Extracted payload bytes, or None if invalid/not found
    """
    H, W, C = image.shape
    mask = (1 << bits_per_channel) - 1

    # Extract all LSBs as bits
    bits = []
    for y in range(H):
        for x in range(W):
            for c in range(C):
                value = image[y, x, c] & mask
                for i in range(bits_per_channel - 1, -1, -1):
                    bits.append((value >> i) & 1)

    # Convert bits to bytes
    data = bytearray()
    for i in range(0, len(bits) - 7, 8):
        byte = 0
        for j in range(8):
            byte = (byte << 1) | bits[i + j]
        data.append(byte)

    # Check magic
    if len(data) < 4:
        return None

    if bytes(data[:2]) != MRP_MAGIC:
        return None

    # Extract length
    length = struct.unpack('>H', bytes(data[2:4]))[0]

    # Validate length
    total_len = 4 + length + 1  # header + payload + crc
    if total_len > len(data):
        return None

    # Extract payload
    payload = bytes(data[4:4 + length])

    # Verify CRC
    stored_crc = data[4 + length]
    computed_crc = crc8(bytes(data[:4 + length]))

    if stored_crc != computed_crc:
        return None

    return payload


# =============================================================================
# Hexagonal Grid Navigation Encoding
# =============================================================================

# Hexagonal wavevectors (unit vectors at 0, 60, -60 degrees)
U_1 = np.array([1.0, 0.0])                    # East (0 degrees)
U_2 = np.array([0.5, math.sqrt(3)/2])         # 60 degrees (NE)
U_3 = np.array([0.5, -math.sqrt(3)/2])        # -60 degrees (SE)

# Stacked as rows for matrix operations
K_HEX = np.vstack([U_1, U_2, U_3])


def position_to_phases(
    position: np.ndarray,
    wavelength: float = 1.0,
    offset: Optional[np.ndarray] = None
) -> Tuple[float, float, float]:
    """
    Convert a 2D position to three phase values using hex wavevectors.

    The three phases represent projections of the position onto the
    three hexagonal wavevector directions, enabling unique position
    encoding with redundancy.

    Args:
        position: 2D position as [x, y]
        wavelength: Spatial wavelength (smaller = higher resolution)
        offset: Optional phase offset [offset_1, offset_2, offset_3]

    Returns:
        Tuple of (phase_1, phase_2, phase_3) in radians [0, 2*pi)
    """
    position = np.asarray(position, dtype=np.float64)

    # Project position onto each wavevector
    projections = K_HEX @ position

    # Convert to phases
    k = TWO_PI / wavelength
    phases = (k * projections) % TWO_PI

    # Apply offset if provided
    if offset is not None:
        offset = np.asarray(offset, dtype=np.float64)
        phases = (phases + offset) % TWO_PI

    return tuple(phases)


def phases_to_position(
    phases: Tuple[float, float, float],
    wavelength: float = 1.0,
    offset: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Convert three phase values back to a 2D position.

    Uses least-squares solution since the system is overdetermined
    (3 equations, 2 unknowns).

    Args:
        phases: Tuple of (phase_1, phase_2, phase_3) in radians
        wavelength: Spatial wavelength used in encoding
        offset: Optional phase offset to subtract

    Returns:
        2D position as [x, y]
    """
    phases = np.asarray(phases, dtype=np.float64)

    # Remove offset if provided
    if offset is not None:
        offset = np.asarray(offset, dtype=np.float64)
        phases = (phases - offset) % TWO_PI

    # Convert phases to projections
    k = TWO_PI / wavelength
    projections = phases / k

    # Solve overdetermined system using least squares
    # K_HEX @ position = projections
    position, _, _, _ = np.linalg.lstsq(K_HEX, projections, rcond=None)

    return position


def encode_trajectory(
    positions: np.ndarray,
    wavelength: float = 1.0
) -> np.ndarray:
    """
    Encode a sequence of positions to RGB values.

    Args:
        positions: Shape (N, 2) array of [x, y] positions
        wavelength: Spatial wavelength

    Returns:
        Shape (N, 3) array of RGB values (uint8)
    """
    positions = np.asarray(positions)
    N = positions.shape[0]

    rgb_trajectory = np.zeros((N, 3), dtype=np.uint8)

    for i in range(N):
        phases = position_to_phases(positions[i], wavelength)
        frame = encode_frame(phases, embed_parity=True)
        rgb_trajectory[i] = frame.rgb

    return rgb_trajectory


def decode_trajectory(
    rgb_trajectory: np.ndarray,
    wavelength: float = 1.0
) -> np.ndarray:
    """
    Decode RGB trajectory back to positions.

    Args:
        rgb_trajectory: Shape (N, 3) array of RGB values
        wavelength: Spatial wavelength used in encoding

    Returns:
        Shape (N, 2) array of [x, y] positions
    """
    rgb_trajectory = np.asarray(rgb_trajectory)
    N = rgb_trajectory.shape[0]

    positions = np.zeros((N, 2), dtype=np.float64)

    for i in range(N):
        rgb = tuple(rgb_trajectory[i])
        phases, _ = decode_frame(rgb, verify=False)
        positions[i] = phases_to_position(phases, wavelength)

    return positions


# =============================================================================
# Verification Utilities
# =============================================================================

def verify_roundtrip(
    phases: Tuple[float, float, float],
    tolerance: float = 0.05
) -> Dict:
    """
    Verify that phases survive encode/decode roundtrip within tolerance.

    Args:
        phases: Original phases to test
        tolerance: Maximum allowed phase error in radians

    Returns:
        Dictionary with verification results:
        - 'success': Overall pass/fail
        - 'phases_in': Input phases
        - 'phases_out': Recovered phases
        - 'errors': Per-channel errors
        - 'max_error': Maximum error across channels
        - 'parity_ok': Parity verification status
    """
    # Encode
    frame = encode_frame(phases, embed_parity=True)

    # Decode
    recovered, parity_ok = decode_frame(frame.rgb, verify=True)

    # Calculate errors (handle phase wrapping)
    errors = []
    for p_in, p_out in zip(phases, recovered):
        # Normalize to [0, 2*pi)
        p_in = p_in % TWO_PI
        p_out = p_out % TWO_PI

        # Calculate minimum circular distance
        diff = abs(p_out - p_in)
        error = min(diff, TWO_PI - diff)
        errors.append(error)

    max_error = max(errors)
    success = max_error <= tolerance and parity_ok

    return {
        'success': success,
        'phases_in': phases,
        'phases_out': recovered,
        'errors': errors,
        'max_error': max_error,
        'parity_ok': parity_ok
    }


# =============================================================================
# Main / Tests
# =============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("MRP-LSB Encoding Module Tests")
    print("=" * 60)

    # Test 1: Basic phase quantization
    print("\n[Test 1] Phase Quantization")
    print("-" * 40)

    test_phases = [0.0, math.pi/4, math.pi/2, math.pi, 3*math.pi/2, TWO_PI - 0.01]
    for phase in test_phases:
        q = quantize_phase(phase, bits=8)
        recovered = dequantize_phase(q, bits=8)
        error = abs(phase - recovered)
        print(f"  Phase: {phase:.4f} rad -> Q: {q:3d} -> "
              f"Recovered: {recovered:.4f} rad (error: {error:.4f})")

    # Test 2: MRP encoding
    print("\n[Test 2] MRP Encoding")
    print("-" * 40)

    for phase in [0.0, 1.5, math.pi, 5.0]:
        mrp = encode_mrp(phase)
        recovered = decode_mrp(mrp)
        coarse_only = decode_coarse_only(mrp)

        print(f"  Phase: {phase:.4f} rad")
        print(f"    Full: {mrp.full:3d} (coarse={mrp.coarse:2d}, fine={mrp.fine:1d})")
        print(f"    Full decode: {recovered:.4f} rad")
        print(f"    Coarse only: {coarse_only:.4f} rad")
        print(f"    Resolution: {mrp.resolution_deg}")

    # Test 3: LSB operations
    print("\n[Test 3] LSB Steganography")
    print("-" * 40)

    original = 0b11110000
    for payload in [0b000, 0b101, 0b111]:
        modified = set_lsb(original, payload, n_bits=3)
        extracted = get_lsb(modified, n_bits=3)
        print(f"  Original: {original:08b}, Payload: {payload:03b}")
        print(f"  Modified: {modified:08b}, Extracted: {extracted:03b}")
        assert extracted == payload, "LSB roundtrip failed!"

    # Test 4: RGB parity
    print("\n[Test 4] RGB Parity")
    print("-" * 40)

    test_rgb = (128, 64, 32)
    parity = compute_rgb_parity(test_rgb[0], test_rgb[1], n_bits=3)
    modified_rgb = (test_rgb[0], test_rgb[1], set_lsb(test_rgb[2], parity, 3))
    verified = verify_rgb_parity(modified_rgb, n_bits=3)
    print(f"  Original RGB: {test_rgb}")
    print(f"  Computed parity: {parity:03b}")
    print(f"  Modified RGB: {modified_rgb}")
    print(f"  Parity verified: {verified}")

    # Test 5: Full frame encoding
    print("\n[Test 5] Frame Encoding")
    print("-" * 40)

    phases = (1.0, 2.5, 4.0)
    frame = encode_frame(phases, embed_parity=True)
    print(f"  Input phases: {phases}")
    print(f"  RGB output: {frame.rgb}")
    print(f"  MRP values: coarse=({frame.mrp[0].coarse}, {frame.mrp[1].coarse}, "
          f"{frame.mrp[2].coarse})")
    print(f"  Parity OK: {frame.parity_ok}")
    print(f"  CRC: 0x{frame.crc:02x}")

    # Test serialization
    frame_bytes = frame.to_bytes()
    print(f"  Serialized: {frame_bytes.hex()}")

    restored = MRPFrame.from_bytes(frame_bytes)
    print(f"  Restored RGB: {restored.rgb}")

    # Test 6: Roundtrip verification
    print("\n[Test 6] Roundtrip Verification")
    print("-" * 40)

    test_cases = [
        (0.0, 0.0, 0.0),
        (math.pi, math.pi, math.pi),
        (1.234, 2.345, 3.456),
        (0.1, 3.14, 6.0),
    ]

    for phases in test_cases:
        result = verify_roundtrip(phases)
        status = "PASS" if result['success'] else "FAIL"
        print(f"  Phases {phases}: {status}")
        print(f"    Max error: {result['max_error']:.6f} rad")
        print(f"    Parity OK: {result['parity_ok']}")

    # Test 7: Hexagonal position encoding
    print("\n[Test 7] Hexagonal Position Encoding")
    print("-" * 40)

    positions = [
        np.array([0.0, 0.0]),
        np.array([1.0, 0.0]),
        np.array([0.5, 0.5]),
        np.array([0.25, 0.75]),
    ]

    wavelength = 1.0
    for pos in positions:
        phases = position_to_phases(pos, wavelength)
        recovered = phases_to_position(phases, wavelength)
        error = np.linalg.norm(pos - recovered)
        print(f"  Position: [{pos[0]:.3f}, {pos[1]:.3f}]")
        print(f"    Phases: ({phases[0]:.3f}, {phases[1]:.3f}, {phases[2]:.3f})")
        print(f"    Recovered: [{recovered[0]:.3f}, {recovered[1]:.3f}]")
        print(f"    Error: {error:.6f}")

    # Test 8: Trajectory encoding
    print("\n[Test 8] Trajectory Encoding")
    print("-" * 40)

    trajectory = np.array([
        [0.0, 0.0],
        [0.1, 0.0],
        [0.2, 0.1],
        [0.3, 0.2],
        [0.4, 0.3],
    ])

    rgb_traj = encode_trajectory(trajectory, wavelength=1.0)
    decoded_traj = decode_trajectory(rgb_traj, wavelength=1.0)

    print(f"  Original trajectory shape: {trajectory.shape}")
    print(f"  RGB trajectory shape: {rgb_traj.shape}")
    print(f"  Decoded trajectory shape: {decoded_traj.shape}")

    total_error = np.sum(np.linalg.norm(trajectory - decoded_traj, axis=1))
    print(f"  Total trajectory error: {total_error:.6f}")

    # Test 9: Image encoding
    print("\n[Test 9] Image Encoding")
    print("-" * 40)

    # Create small test phase image
    H, W = 4, 4
    phases_image = np.random.uniform(0, TWO_PI, (H, W, 3)).astype(np.float32)

    # Encode and decode
    rgb_image = encode_phase_image(phases_image, embed_parity=True)
    decoded_phases, parity_mask = decode_phase_image(rgb_image, verify=True)

    print(f"  Phase image shape: {phases_image.shape}")
    print(f"  RGB image shape: {rgb_image.shape}")
    print(f"  Decoded shape: {decoded_phases.shape}")
    print(f"  Parity OK percentage: {100 * parity_mask.mean():.1f}%")

    # Calculate max error
    errors = np.minimum(
        np.abs(decoded_phases - phases_image),
        TWO_PI - np.abs(decoded_phases - phases_image)
    )
    print(f"  Max phase error: {errors.max():.6f} rad")

    # Test 10: Payload steganography
    print("\n[Test 10] Payload Steganography")
    print("-" * 40)

    # Create test image
    test_image = np.random.randint(0, 256, (16, 16, 3), dtype=np.uint8)

    # Test payloads
    payloads = [
        b"Hello",
        b"MRP-LSB Test!",
        bytes(range(32)),
    ]

    for payload in payloads:
        embedded = embed_payload_in_image(test_image, payload, bits_per_channel=2)
        extracted = extract_payload_from_image(embedded, bits_per_channel=2)

        match = extracted == payload
        print(f"  Payload: {payload[:20]}{'...' if len(payload) > 20 else ''}")
        print(f"    Length: {len(payload)} bytes")
        print(f"    Extracted: {extracted[:20] if extracted else None}"
              f"{'...' if extracted and len(extracted) > 20 else ''}")
        print(f"    Match: {match}")

    # Test 11: CRC-8
    print("\n[Test 11] CRC-8 Checksum")
    print("-" * 40)

    test_data = [
        b"",
        b"\x00",
        b"123456789",
        b"Hello World",
    ]

    for data in test_data:
        checksum = crc8(data)
        print(f"  Data: {data!r}")
        print(f"    CRC-8: 0x{checksum:02x}")

    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)
