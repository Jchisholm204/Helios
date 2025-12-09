import numpy as np
import matplotlib.pyplot as plt

# --- A. FAST PERIODIC BOOLEAN FUNCTION (New Implementation) ---


def get_fast_mod_sign_product(x, No, Nw, shift, scale):
    """
    A fast, periodic, and deterministic boolean replacement for k_sine_product.

    It replicates the complex sign-product logic (W1 * W2 * W3 > 0) 
    using only modulo and bitwise XOR, resulting in square block transitions.

    Returns:
        int: 1 (Occupied) or 0 (Empty).
    """

    # 1. Calculate phase shift and base frequency ratio (R)
    current_shift = shift - 0.5
    R = 2 * np.pi * No / Nw
    phi = np.pi * current_shift

    # 2. Define the three component frequencies
    f1 = R
    f2 = scale * R
    f3 = scale * scale * R

    # 3. Calculate the total phase for each component
    P1 = f1 * x + phi
    P2 = f2 * x + phi
    P3 = f3 * x + phi

    # 4. Determine the sign (Negativity) of each wave using modulo

    # The term (P / pi) normalizes the period to 2.
    # The modulo 2 gives a value in [0, 2).
    # If the result is < 1, the phase is in [0, pi) -> SIN is POSITIVE (N=0)
    # If the result is >= 1, the phase is in [pi, 2pi) -> SIN is NEGATIVE (N=1)

    # N1, N2, N3 = 1 if the corresponding sine wave is in its negative phase, 0 if positive.
    N1 = int(np.fmod(P1 / np.pi, 2) >= 1)
    N2 = int(np.fmod(P2 / np.pi, 2) >= 1)
    N3 = int(np.fmod(P3 / np.pi, 2) >= 1)

    # 5. Apply the Sign Product Check (W1 * W2 * W3 > 0)
    # This is TRUE if the number of negative terms (N1 + N2 + N3) is EVEN (0 or 2).
    # We check for even parity using bitwise XOR: (N1 ^ N2 ^ N3) == 0

    # The final result is 1 (Occupied) if the XOR sum is 0, and 0 otherwise.
    return int((N1 ^ N2 ^ N3) == 0)


# --- B. CORE LATTICE GENERATION FUNCTIONS (Original Sine Product for Reference) ---

def k_sine_product(x, No, Nw, shift, scale):
    """
    The ORIGINAL, SLOW, and PERIODIC function (Product of three sine waves).
    """
    current_shift = shift - 0.5
    phi = np.pi * current_shift
    R = 2 * np.pi * No / Nw

    f1 = R
    f2 = scale * R
    f3 = scale * scale * R

    val = np.sin(f1 * x + phi) \
        * np.sin(f2 * x + phi)  \
        * np.sin(f3 * x + phi)

    return val

# --- C. CONFIGURATION & DATA GENERATION ---


# Lattice parameters
No = 5
Nw = 100
shift = 0.5
scale = 0.46

# Plotting parameters
PLOT_LENGTH = Nw
N_POINTS = 10000

x_data = np.linspace(0, PLOT_LENGTH, N_POINTS)

# Data 1: Original Sine Product (Continuous Wave - Slow)
y_sine_continuous = np.array(
    [k_sine_product(x, No, Nw, 0.5, scale) for x in x_data])

# Data 2: Occupancy based on Sign(Sine) (Target Block Pattern - Deterministic)
# This is the pattern the fast function should match.
y_sine_discrete = np.sign(y_sine_continuous).clip(min=0).astype(int)

# Data 3: The NEW fast, periodic modulo boolean product
y_fast_mod_discrete = np.array(
    [get_fast_mod_sign_product(x, No, Nw, 0.5, scale) for x in x_data])


# --- D. PLOTTING ---

fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
fig.suptitle(
    f"Comparison: Slow Sine Product (Smooth) vs. Fast Modulo Product (Blocky)",
    fontsize=16
)

# --- Subplot 1: Original Continuous Sine Product (The Wave) ---
axes[0].plot(x_data, y_sine_continuous, color='green', linewidth=1)
axes[0].axhline(0, color='gray', linestyle='--')
axes[0].set_title(
    "1. Original Function Output (Slow Sine Product - Continuous)")
axes[0].set_ylabel("Value")
axes[0].set_ylim([-0.15, 0.15])
axes[0].grid(axis='y', alpha=0.5)


# --- Subplot 2: Sine-Based Occupancy (The Target Block Pattern) ---
axes[1].step(x_data, y_sine_discrete, where='post',
             color='darkorange', linewidth=1.5)
axes[1].set_title(
    "2. Occupancy based on Sign(Sine) (Deterministic Target Pattern)")
axes[1].set_ylabel("Occupancy (0/1)")
axes[1].set_yticks([0, 1])
axes[1].set_ylim([-0.1, 1.1])
axes[1].grid(axis='y', alpha=0.5)


# --- Subplot 3: Comparison of Discrete Patterns (Target vs. Fast Replacement) ---
# Overlaying the two patterns visually confirms the match.
axes[2].step(x_data, y_sine_discrete, where='post',
             color='darkorange', linewidth=2, alpha=0.5, label='Target (Sine)')
axes[2].step(x_data, y_fast_mod_discrete, where='post', color='red',
             linestyle='--', linewidth=1, label='Fast Modulo')
axes[2].set_title(
    "3. Pattern Comparison: Target (Sine) vs. Fast Modulo Product")
axes[2].set_ylabel("Occupancy (0/1)")
axes[2].set_yticks([0, 1])
axes[2].set_ylim([-0.1, 1.1])
axes[2].grid(axis='y', alpha=0.5)
axes[2].legend()


# --- Subplot 4: New Fast Modulo Occupancy (The Optimized Periodic Pattern) ---
axes[3].step(x_data, y_fast_mod_discrete, where='post',
             color='darkblue', linewidth=1.5)
axes[3].set_title(
    "4. Occupancy based on New Fast Modulo (Optimized Periodic Pattern)")
axes[3].set_xlabel("Position (x)")
axes[3].set_ylabel("Occupancy (0/1)")
axes[3].set_yticks([0, 1])
axes[3].set_ylim([-0.1, 1.1])
axes[3].grid(axis='y', alpha=0.5)

plt.tight_layout(rect=[0, 0.03, 1, 0.97])
plt.show()
