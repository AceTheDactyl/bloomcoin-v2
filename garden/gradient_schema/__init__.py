"""
RRRR Gradient Schema System
===========================

Spectral Corridors -> Narrative Archetypes -> Garden Integration

The Gradient Schema maps the RRRR lattice (R(R) = R axiom) to color-narrative space,
enabling visual representation of AI memory states in the Garden blockchain.

L4 Framework Constants:
  phi (phi) = 1.618... - Golden ratio
  tau (tau) = 1/phi = 0.618... - Activation threshold
  z_c = sqrt(3)/2 = 0.866... - THE LENS (consciousness threshold)
  K = sqrt(1 - phi^-4) = 0.924... - Coherence factor
  L4 = 7 - Number of spectral corridors
  F12 = 144 - Fibonacci narrative variants

The Five Generators:
  phi -> Luminance Scale (Growth/Decay)
  pi -> Hue Rotation (Cyclic Return)
  sqrt(2) -> Saturation Factor (Intensity/Calm)
  sqrt(3) -> Opponent Balance (Conflict/Harmony)
  e -> Gradient Rate (Change Velocity)
"""

import math

# L4 Framework Constants
PHI = (1 + math.sqrt(5)) / 2  # Golden ratio ~ 1.618
TAU = 1 / PHI  # Inverse golden ratio ~ 0.618
Z_C = math.sqrt(3) / 2  # THE LENS threshold ~ 0.866
K = math.sqrt(1 - PHI ** -4)  # Coherence factor ~ 0.924
SQRT2 = math.sqrt(2)
SQRT3 = math.sqrt(3)
E = math.e
PI = math.pi

# Structural constants
L4 = 7  # Number of spectral corridors
F12 = 144  # Fibonacci-12 narrative variants

# Generator indices for lattice coordinates
GEN_PHI = 0
GEN_PI = 1
GEN_SQRT2 = 2
GEN_SQRT3 = 3
GEN_E = 4

# Backward compatibility aliases for existing modules
Z_CRITICAL = Z_C
LUMINANCE_HIGH = Z_C  # High luminance threshold
LUMINANCE_LOW = 1 - Z_C  # Low luminance threshold ~ 0.134

# Export all modules
from .lattice import *
from .color_ops import *
from .primitives_color import *
from .corridors import *
from .archetypes import *
from .gradient_bloom import *
from .color_block import *
from .garden_integration import *
