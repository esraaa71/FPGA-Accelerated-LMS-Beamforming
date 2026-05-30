# ============================================================
# FINAL PYNQ Jupyter Code - HARDWARE ONLY 4-Tap LMS
# ============================================================
# Hardware path:
#   Python -> AXI DMA MM2S -> LMS FPGA IP -> AXI DMA S2MM -> Python
#
# LMS calculation is done ONLY inside FPGA hardware.
# No software LMS model.
# No ARM LMS loop.
# ============================================================

from pynq import Overlay, allocate
import numpy as np
import matplotlib.pyplot as plt
import time
import os

# ============================================================
# Configuration
# ============================================================

BITFILE = "./design_112.bit"

CHUNK_SIZE   = 1024
NUM_CHUNKS   = 300
TOTAL_SAMPLES = CHUNK_SIZE * NUM_CHUNKS

PIPE_LATENCY = 7
Q15          = 32768.0

# Signal parameters
SAMP_RATE    = 140e6
F_PILOT      = 1e6
PILOT_AMP    = 0.30

# Hardware LMS target channel
CHANNEL_GAIN = 0.6 + 0.3j
NOISE_AMP    = 0.02

# LMS step size
MU_FLOAT     = 0.003
MU_Q15       = int(MU_FLOAT * Q15)

LOG_EVERY = 30
SAVE_PLOT = True

# ============================================================
# Header
# ============================================================

print("=" * 72)
print("FINAL HARDWARE ONLY 4-Tap Complex LMS on PYNQ")
print("=" * 72)

print("\nConfiguration:")
print("Bitfile        :", BITFILE)
print("Chunk size     :", CHUNK_SIZE)
print("Chunks         :", NUM_CHUNKS)
print("Total samples  :", TOTAL_SAMPLES)
print("Sample rate    :", SAMP_RATE)
print("Pilot freq     :", F_PILOT)
print("Pilot amp      :", PILOT_AMP)
print("Channel gain   :", CHANNEL_GAIN)
print("Noise amp      :", NOISE_AMP)
print("mu float       :", MU_FLOAT)
print("mu Q1.15       :", MU_Q15)
print("Pipeline delay :", PIPE_LATENCY)

# ============================================================
# Load FPGA overlay
# ============================================================

print("\n[1] Loading FPGA overlay...")

if not os.path.exists(BITFILE):
    raise FileNotFoundError(
        "Bitfile not found: {}. Put design_112.bit in the same folder as this notebook.".format(BITFILE)
    )

overlay = Overlay(BITFILE)

print("Overlay loaded.")

print("\nAvailable IPs:")
for ip_name in overlay.ip_dict.keys():
    print("  ", ip_name)

# These names must match your Vivado block design
dma = overlay.axi_dma_0
gpio_mu = overlay.axi_gpio_0.channel1

gpio_mu.write(MU_Q15, 0xFFFF)

print("\nDMA found       : axi_dma_0")
print("GPIO mu found  : axi_gpio_0.channel1")
print("mu written     : {} -> Q1.15 {}".format(MU_FLOAT, MU_Q15))

# ============================================================
# Allocate DMA buffers
# ============================================================

print("\n[2] Allocating DMA buffers...")

in_buf  = allocate(shape=(CHUNK_SIZE,), dtype=np.uint64)
out_buf = allocate(shape=(CHUNK_SIZE,), dtype=np.uint64)

print("Input buffer  :", in_buf.shape, in_buf.dtype)
print("Output buffer :", out_buf.shape, out_buf.dtype)

# ============================================================
# Q1.15 helpers
# ============================================================

def to_q15(arr):
    """
    Convert float array to signed Q1.15 int16.
    Range: [-1.0, 1.0)
    """
    arr = np.asarray(arr, dtype=np.float32)
    arr = np.clip(arr, -1.0, 1.0 - 1.0 / Q15)
    return np.round(arr * Q15).astype(np.int16)


def from_q15_signed(u16_arr):
    """
    Convert uint16 bits to signed Q1.15 float.
    """
    return u16_arr.view(np.int16).astype(np.float32) / Q15


def pack_word_64(x_re, x_im, d_re, d_im):
    """
    Pack one sample into 64-bit AXI Stream word.

    Input format expected by hardware:
      tdata[63:48] = x_re
      tdata[47:32] = x_im
      tdata[31:16] = d_re
      tdata[15:0]  = d_im

    All values are Q1.15 signed.
    """
    xr = to_q15(x_re).view(np.uint16).astype(np.uint64)
    xi = to_q15(x_im).view(np.uint16).astype(np.uint64)
    dr = to_q15(d_re).view(np.uint16).astype(np.uint64)
    di = to_q15(d_im).view(np.uint16).astype(np.uint64)

    return (xr << 48) | (xi << 32) | (dr << 16) | di


def unpack_output(result_u64):
    """
    Unpack hardware output.

    Output format from hardware:
      tdata[63:48] = y_re
      tdata[47:32] = y_im
      tdata[31:16] = e_re
      tdata[15:0]  = e_im

    Returns:
      y_complex, e_complex
    """
    result_u64 = result_u64.astype(np.uint64)

    yr_u16 = ((result_u64 >> 48) & 0xFFFF).astype(np.uint16)
    yi_u16 = ((result_u64 >> 32) & 0xFFFF).astype(np.uint16)
    er_u16 = ((result_u64 >> 16) & 0xFFFF).astype(np.uint16)
    ei_u16 = ( result_u64        & 0xFFFF).astype(np.uint16)

    y_re = from_q15_signed(yr_u16)
    y_im = from_q15_signed(yi_u16)
    e_re = from_q15_signed(er_u16)
    e_im = from_q15_signed(ei_u16)

    y = y_re + 1j * y_im
    e = e_re + 1j * e_im

    return y.astype(np.complex64), e.astype(np.complex64)


def dma_transfer(words_in):
    """
    Send packed input samples to hardware LMS and receive packed output.
    """
    in_buf[:] = words_in
    out_buf[:] = 0

    dma.recvchannel.transfer(out_buf)
    dma.sendchannel.transfer(in_buf)

    dma.sendchannel.wait()
    dma.recvchannel.wait()

    return out_buf[:].copy()

# ============================================================
# Signal generator
# ============================================================

class ComplexSignalGenerator:
    def __init__(self, seed=42):
        self.idx = 0
        self.rng = np.random.RandomState(seed)

    def next_chunk(self, n):
        t = np.arange(self.idx, self.idx + n, dtype=np.float64) / SAMP_RATE

        x = PILOT_AMP * np.exp(1j * 2.0 * np.pi * F_PILOT * t)

        noise = NOISE_AMP * (
            self.rng.randn(n).astype(np.float32)
            + 1j * self.rng.randn(n).astype(np.float32)
        )

        d = CHANNEL_GAIN * x + noise

        self.idx += n

        return x.astype(np.complex64), d.astype(np.complex64)

# ============================================================
# Hardware streaming loop
# ============================================================

print("\n[3] Starting HARDWARE DMA streaming...")
print("Streaming {} chunks x {} samples = {:,} samples".format(
    NUM_CHUNKS, CHUNK_SIZE, TOTAL_SAMPLES
))
print()

gen = ComplexSignalGenerator(seed=42)

all_err_hw = []
all_snr_hw = []
all_time_ms = []
hw_times_ms = []

last_x = None
last_d = None
last_y = None
last_e = None

t_total_start = time.time()

for ci in range(NUM_CHUNKS):
    x, d = gen.next_chunk(CHUNK_SIZE)

    words_in = pack_word_64(x.real, x.imag, d.real, d.imag)

    t0 = time.perf_counter()

    result_words = dma_transfer(words_in)

    hw_time = time.perf_counter() - t0
    hw_times_ms.append(hw_time * 1000.0)

    y_hw, e_hw = unpack_output(result_words)

    # Ignore invalid first samples due to hardware pipeline latency
    e_valid = e_hw[PIPE_LATENCY:]
    d_valid = d[PIPE_LATENCY:]
    y_valid = y_hw[PIPE_LATENCY:]

    err_rms = float(np.sqrt(np.mean(np.abs(e_valid) ** 2)))
    sig_rms = float(np.sqrt(np.mean(np.abs(d_valid) ** 2)))
    snr_hw = 10.0 * np.log10((sig_rms ** 2) / (err_rms ** 2 + 1e-12))

    all_err_hw.append(err_rms)
    all_snr_hw.append(snr_hw)
    all_time_ms.append(ci * CHUNK_SIZE / SAMP_RATE * 1e3)

    last_x = x.copy()
    last_d = d.copy()
    last_y = y_hw.copy()
    last_e = e_hw.copy()

    if ci == 0 or ci % LOG_EVERY == 0 or ci == NUM_CHUNKS - 1:
        elapsed = time.time() - t_total_start
        throughput = ((ci + 1) * CHUNK_SIZE) / elapsed / 1e6

        current_reduction = (
            1.0 - all_err_hw[-1] / (all_err_hw[0] + 1e-12)
        ) * 100.0

        print(
            "Chunk {:>4}/{} | HW e_rms={:.5f} | reduction={:.1f}% | "
            "SNR={:+.2f} dB | HW time={:.3f} ms/chunk | tput={:.3f} MSps | {:.2f}s".format(
                ci + 1,
                NUM_CHUNKS,
                err_rms,
                current_reduction,
                snr_hw,
                np.mean(hw_times_ms),
                throughput,
                elapsed
            )
        )

t_total = time.time() - t_total_start

# ============================================================
# Results
# ============================================================

all_err_hw = np.array(all_err_hw, dtype=np.float64)
all_snr_hw = np.array(all_snr_hw, dtype=np.float64)
all_time_ms = np.array(all_time_ms, dtype=np.float64)
hw_times_ms = np.array(hw_times_ms, dtype=np.float64)

noise_floor = float(np.sqrt(2.0 * NOISE_AMP ** 2))

hw_error_start = float(all_err_hw[0])
hw_error_end = float(all_err_hw[-1])
hw_reduction = (1.0 - hw_error_end / (hw_error_start + 1e-12)) * 100.0

avg_hw_time_ms = float(np.mean(hw_times_ms))
throughput_total = TOTAL_SAMPLES / t_total / 1e6 if t_total > 0 else 0.0

print("\n" + "=" * 72)
print("FINAL HARDWARE ONLY RESULTS")
print("=" * 72)

print("Total samples        : {:,}".format(TOTAL_SAMPLES))
print("Total time           : {:.6f} s".format(t_total))
print("Throughput total     : {:.4f} MSamples/s".format(throughput_total))

print("\nHardware timing:")
print("Average HW DMA+LMS   : {:.3f} ms/chunk".format(avg_hw_time_ms))
print("Min HW DMA+LMS       : {:.3f} ms/chunk".format(float(np.min(hw_times_ms))))
print("Max HW DMA+LMS       : {:.3f} ms/chunk".format(float(np.max(hw_times_ms))))

print("\nHardware LMS metrics:")
print("HW Error RMS start   : {:.5f}".format(hw_error_start))
print("HW Error RMS end     : {:.5f}".format(hw_error_end))
print("HW Error reduction   : {:.1f}%".format(hw_reduction))
print("HW Final SNR         : {:.2f} dB".format(float(all_snr_hw[-1])))
print("Noise floor approx   : {:.5f}".format(noise_floor))

# ============================================================
# Save capture
# ============================================================

np.savez(
    "hardware_only_lms_results.npz",
    err_rms_hw=all_err_hw,
    snr_hw=all_snr_hw,
    time_ms=all_time_ms,
    hw_times_ms=hw_times_ms,
    last_x=last_x,
    last_d=last_d,
    last_y=last_y,
    last_e=last_e,
    mu_float=MU_FLOAT,
    mu_q15=MU_Q15,
    chunk_size=CHUNK_SIZE,
    num_chunks=NUM_CHUNKS,
    pipe_latency=PIPE_LATENCY,
)

print("\nSaved data: hardware_only_lms_results.npz")

# ============================================================
# Plot
# ============================================================

if SAVE_PLOT:
    print("\n[4] Plotting...")

    plt.figure(figsize=(13, 10))

    plt.subplot(4, 1, 1)
    plt.plot(all_time_ms, all_err_hw, label="Hardware Error RMS")
    plt.axhline(noise_floor, linestyle="--", label="Noise floor approx")
    plt.title("Hardware Only 4-Tap LMS - Error RMS")
    plt.xlabel("Time ms")
    plt.ylabel("Error RMS")
    plt.grid(True)
    plt.legend()

    plt.subplot(4, 1, 2)
    plt.plot(all_time_ms, all_snr_hw, label="Hardware SNR")
    plt.title("Hardware Only 4-Tap LMS - SNR")
    plt.xlabel("Time ms")
    plt.ylabel("SNR dB")
    plt.grid(True)
    plt.legend()

    plt.subplot(4, 1, 3)
    plt.plot(all_time_ms, hw_times_ms, label="DMA + Hardware LMS time")
    plt.title("Hardware Processing Time per Chunk")
    plt.xlabel("Time ms")
    plt.ylabel("ms/chunk")
    plt.grid(True)
    plt.legend()

    plt.subplot(4, 1, 4)
    nshow = min(256, len(last_x))
    t_us = np.arange(nshow) / SAMP_RATE * 1e6

    plt.plot(t_us, last_x[:nshow].real, label="x real")
    plt.plot(t_us, last_d[:nshow].real, label="d real")
    plt.plot(t_us, last_y[:nshow].real, label="y hardware real")
    plt.title("Final Chunk Waveform Snapshot")
    plt.xlabel("Time us")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig("hardware_only_lms_results.png", dpi=150)
    plt.show()

    print("Saved plot: hardware_only_lms_results.png")

# ============================================================
# Cleanup
# ============================================================

in_buf.freebuffer()
out_buf.freebuffer()

print("\nBuffers freed.")
print("Done.")