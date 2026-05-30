"""
single_tap.py
=====================
Hardware Execution Script for Single-Tap I/Q LMS Filter on PYNQ

Runs the test scenario (generating a complex wideband pilot passing through 
a multipath channel with noise), but pushes the data through the FPGA Programmable 
Logic (PL) using the `single_tap_iq.bit` overlay via AXI DMA.
"""

import pynq
import numpy as np
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============================================================
# Configuration
# ============================================================
# Note: Ensure "single_tap_iq.bit" and "single_tap_iq.hwh" are in the same directory!
BITSTREAM     = "single_tap_iq.bit"

CHUNK_SIZE    = 1024
NUM_CHUNKS    = 300
Q15           = 32768.0

H_CHANNEL     = np.array([0.60, 0.50, -0.40, 0.30], dtype=np.complex64)
PILOT_FREQS   = [1e6, 7e6, 18e6, 35e6]
PILOT_AMP     = 0.30          
INTERF_FREQ   = 45e6
INTERF_AMP    = 0.02          
NOISE_AMP     = 0.005         
SAMP_RATE     = 140e6

MU_FLOAT      = 0.005         

# Theoretical floors for comparison
NOISE_FLOOR   = float(np.sqrt(2 * INTERF_AMP**2 * 0.5 + 2 * NOISE_AMP**2))
ISI_POWER     = float(np.sum(np.abs(H_CHANNEL[1:])**2) * PILOT_AMP**2)
ISI_FLOOR     = float(np.sqrt(ISI_POWER + 2 * INTERF_AMP**2 * 0.5 + 2 * NOISE_AMP**2))

def float_to_q15(x):
    """Convert float array to Q15 int16 format, with saturation."""
    x = np.clip(x, -1.0, 1.0 - (1/32768.0))
    return np.round(x * 32768.0).astype(np.int16)

def q15_to_float(x):
    """Convert Q15 int16 array back to float."""
    return x.astype(np.float32) / 32768.0

# ============================================================
# Complex I/Q Signal Generator
# ============================================================
class SignalGenerator:
    def __init__(self, seed=42):
        self.idx = 0
        self.rng = np.random.default_rng(seed)

    def next_chunk(self, n):
        t = np.arange(self.idx, self.idx + n, dtype=np.float64) / SAMP_RATE

        # Complex pilot
        pilot = np.zeros(n, dtype=np.complex64)
        for f in PILOT_FREQS:
            pilot += (PILOT_AMP / len(PILOT_FREQS)) * np.exp(1j * 2 * np.pi * f * t).astype(np.complex64)

        # Multipath channel output
        d = np.convolve(pilot, H_CHANNEL, mode='full')[:n].astype(np.complex64)

        # Interference + AWGN
        interf = (INTERF_AMP * np.exp(1j * 2 * np.pi * INTERF_FREQ * t)).astype(np.complex64)
        noise  = (NOISE_AMP * (self.rng.standard_normal(n) + 1j * self.rng.standard_normal(n)) / np.sqrt(2)).astype(np.complex64)
        d = d + interf + noise

        self.idx += n
        return pilot, d   # x is pilot, d is channel+noise

# ============================================================
# Hardware Execution
# ============================================================
print(f"Loading Overlay: {BITSTREAM}...")
try:
    overlay = pynq.Overlay(BITSTREAM)
except Exception as e:
    print(f"ERROR: Failed to load {BITSTREAM}. Make sure the .bit and .hwh files exist and the names match!")
    raise e

# Setup DMA and GPIO
# Note: Ensure these names match your Vivado Block Design
dma = overlay.axi_dma_0   
gpio = overlay.axi_gpio_0 

# Write step size (mu) to hardware via AXI GPIO
mu_q15 = int(float_to_q15(MU_FLOAT))
# Interpret as unsigned 16-bit for AXI GPIO write
if mu_q15 < 0:
    mu_q15 += 65536
gpio.channel1.write(mu_q15, 0x1) 
print(f"Set step-size mu = {MU_FLOAT} (Q15: {mu_q15})")

# Allocate PYNQ contiguous memory buffers (64-bit uint64 arrays)
in_buf = pynq.allocate(shape=(CHUNK_SIZE,), dtype=np.uint64)
out_buf = pynq.allocate(shape=(CHUNK_SIZE,), dtype=np.uint64)

gen = SignalGenerator(seed=42)

all_err = []
all_snr = []
all_time = []
hw_times = []

print("\nStarting Hardware Streaming...")
t_total_start = time.time()

for ci in range(NUM_CHUNKS):
    x_in, d_in = gen.next_chunk(CHUNK_SIZE)
    
    # 1. Quantize to Q15 (int16)
    xr_q15 = float_to_q15(x_in.real)
    xi_q15 = float_to_q15(x_in.imag)
    dr_q15 = float_to_q15(d_in.real)
    di_q15 = float_to_q15(d_in.imag)

    # Convert to uint16 to safely shift bits
    xr_u16 = xr_q15.view(np.uint16).astype(np.uint64)
    xi_u16 = xi_q15.view(np.uint16).astype(np.uint64)
    dr_u16 = dr_q15.view(np.uint16).astype(np.uint64)
    di_u16 = di_q15.view(np.uint16).astype(np.uint64)

    # 2. Pack 64-bit word: {xr[63:48], xi[47:32], dr[31:16], di[15:0]}
    in_buf[:] = (xr_u16 << 48) | (xi_u16 << 32) | (dr_u16 << 16) | di_u16

    # 3. Fire DMA Transfer
    t0 = time.perf_counter()
    dma.recvchannel.transfer(out_buf)
    dma.sendchannel.transfer(in_buf)
    dma.sendchannel.wait()
    dma.recvchannel.wait()
    hw_times.append(time.perf_counter() - t0)

    # 4. Unpack 64-bit output: {yr[63:48], yi[47:32], er[31:16], ei[15:0]}
    # Convert buffer view to standard numpy array before shifting
    out_arr = np.array(out_buf, dtype=np.uint64)
    yr_u16 = (out_arr >> 48) & 0xFFFF
    yi_u16 = (out_arr >> 32) & 0xFFFF
    er_u16 = (out_arr >> 16) & 0xFFFF
    ei_u16 = (out_arr      ) & 0xFFFF

    # Interpret as signed 16-bit
    yr_q15 = yr_u16.astype(np.uint16).view(np.int16)
    yi_q15 = yi_u16.astype(np.uint16).view(np.int16)
    er_q15 = er_u16.astype(np.uint16).view(np.int16)
    ei_q15 = ei_u16.astype(np.uint16).view(np.int16)

    # 5. Convert back to float for analysis
    er_f = q15_to_float(er_q15)
    ei_f = q15_to_float(ei_q15)
    
    # Calculate exact SNR using quantized inputs
    dr_f = q15_to_float(dr_q15)  
    di_f = q15_to_float(di_q15)

    # 6. Calculate Metrics
    e_sq_sum = np.sum(er_f**2 + ei_f**2)
    d_sq_sum = np.sum(dr_f**2 + di_f**2)
    
    err_rms = np.sqrt(e_sq_sum / CHUNK_SIZE)
    sig_rms = np.sqrt(d_sq_sum / CHUNK_SIZE)
    snr_db  = 10 * np.log10(sig_rms**2 / (err_rms**2 + 1e-12))
    
    all_err.append(err_rms)
    all_snr.append(snr_db)
    all_time.append(ci * CHUNK_SIZE / SAMP_RATE * 1e3)

    if ci % 30 == 0 or ci == NUM_CHUNKS - 1:
        elapsed = time.time() - t_total_start
        print(f"  Chunk {ci+1:>4}/{NUM_CHUNKS} | e_rms={err_rms:.4f} | SNR={snr_db:+.1f} dB | {elapsed:.2f}s")

# Clean up
in_buf.close()
out_buf.close()

# ============================================================
# Results Summary
# ============================================================
red = (1 - all_err[-1] / (all_err[0] + 1e-12)) * 100
total_hw_time = np.sum(hw_times)

print("\n" + "=" * 60)
print("  HARDWARE RESULTS (1-Tap Complex I/Q LMS)")
print("=" * 60)
print(f"  Error RMS (start): {all_err[0]:.5f}")
print(f"  Error RMS (end):   {all_err[-1]:.5f}")
print(f"  Noise floor:       {NOISE_FLOOR:.5f}")
print(f"  ISI floor:         {ISI_FLOOR:.5f} (Expected for 1-Tap)")
print(f"  Error Reduction:   {red:.1f}%")
print(f"  Final SNR:         {all_snr[-1]:.1f} dB")
print(f"  HW Processing Time:{total_hw_time:.6f} seconds for {NUM_CHUNKS} chunks")
print("=" * 60)

# ============================================================
# Plot
# ============================================================
print("\nPlotting...")
fig, ax1 = plt.subplots(figsize=(10, 5))
ax1.plot(all_time, all_err, color='#ffa726', lw=2, label=f'Hardware 1-Tap I/Q (end={all_err[-1]:.4f})')
ax1.axhline(ISI_FLOOR, color='#ef5350', ls='--', label=f'Theoretical ISI Floor ({ISI_FLOOR:.4f})')
ax1.axhline(NOISE_FLOOR, color='#66bb6a', ls=':', label=f'Absolute Noise Floor ({NOISE_FLOOR:.4f})')
ax1.fill_between(all_time, all_err, NOISE_FLOOR, alpha=0.1, color='#ffa726')

ax1.set_xlabel("Time (ms)", fontweight='bold')
ax1.set_ylabel("Error RMS", fontweight='bold')
ax1.set_title("Hardware 1-Tap I/Q LMS Convergence (PYNQ execution)", fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend()

plt.tight_layout()
plt.savefig('single_tap_iq_hardware_convergence.png', dpi=150)
print("  ✓ Saved plot to 'single_tap_iq_hardware_convergence.png'")
