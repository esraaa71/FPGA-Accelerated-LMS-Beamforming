"""
1-Tap LMS Software-Only Test
============================

No FPGA.
No DMA.
No GNU Radio.
No UDP.

This script runs the same test scenario as the 4-tap software/hardware tests,
but the adaptive LMS filter itself is 1-tap only.

It generates:
  x(n): pilot/reference signal
  d(n): 4-tap multipath channel output + interference + noise

Then 1-tap software LMS computes:
  y(n) = w0 * x(n)
  e(n) = d(n) - y(n)
  w0   = w0 + mu * e(n) * conj(x(n))

Outputs:
  - chunk readings
  - Error RMS
  - SNR
  - processing time
  - throughput
  - final learned weight
  - plot

Important:
  The channel is still 4-tap:
    h = [0.60, 0.50, -0.40, 0.30]

  But the LMS is only 1-tap.
  So it cannot fully remove multipath ISI.
  It should perform worse than the 4-tap LMS.
"""

import numpy as np
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# Configuration
# ============================================================

CHUNK_SIZE = 1024
NUM_CHUNKS = 300

# The real channel has 4 taps
CHANNEL_TAPS = 4

# But the adaptive LMS filter has only 1 tap
LMS_TAPS = 1

Q15 = 32768.0

MU_FLOAT = 0.005

USE_COMPLEX_IQ = False

H_CHANNEL = np.array([0.60, 0.50, -0.40, 0.30], dtype=np.complex64)

SAMP_RATE = 100e6
PILOT_FREQS = [1e6, 7e6, 18e6, 35e6]
PILOT_AMP = 0.30

INTERF_FREQ = 45e6
INTERF_AMP = 0.02
NOISE_AMP = 0.005

LOG_EVERY = 30

# Keep True to mimic Q1.15 numerical behavior like FPGA input/output
EMULATE_Q15_INPUT = True


# ============================================================
# Q1.15 helpers
# ============================================================

def to_q15_float(arr):
    """
    Quantize float to Q1.15 then convert back to float.
    This mimics the FPGA numerical range.
    """
    arr = np.asarray(arr, dtype=np.float32)
    arr = np.clip(arr, -1.0, 1.0 - 1.0 / Q15)
    q = np.round(arr * Q15).astype(np.int16)
    return q.astype(np.float32) / Q15


def quantize_complex_q15(x):
    """
    Apply Q1.15 quantization to real and imaginary parts separately.
    """
    xr = to_q15_float(x.real)
    xi = to_q15_float(x.imag)
    return (xr + 1j * xi).astype(np.complex64)


# ============================================================
# Signal generator
# ============================================================

class SignalGenerator:
    """
    Generates continuous x(n) and d(n).

    x(n): pilot/reference
    d(n): 4-tap multipath channel output + interference + noise
    """

    def __init__(self, seed=42):
        self.idx = 0
        self.rng = np.random.RandomState(seed)

        # previous x samples for continuous 4-tap channel
        self.prev_x = np.zeros(CHANNEL_TAPS - 1, dtype=np.complex64)

    def next_chunk(self, n):
        t = np.arange(self.idx, self.idx + n, dtype=np.float64) / SAMP_RATE

        # Generate pilot x
        x = np.zeros(n, dtype=np.complex64)

        for f in PILOT_FREQS:
            if USE_COMPLEX_IQ:
                tone = np.exp(1j * 2.0 * np.pi * f * t).astype(np.complex64)
            else:
                tone = np.cos(2.0 * np.pi * f * t).astype(np.float32).astype(np.complex64)

            x += (PILOT_AMP / len(PILOT_FREQS)) * tone

        # Continuous 4-tap channel:
        # d[n] = h0*x[n] + h1*x[n-1] + h2*x[n-2] + h3*x[n-3]
        x_hist = np.concatenate([self.prev_x, x]).astype(np.complex64)

        conv = np.convolve(x_hist, H_CHANNEL, mode="full").astype(np.complex64)
        d = conv[CHANNEL_TAPS - 1: CHANNEL_TAPS - 1 + n].copy()

        self.prev_x = x_hist[-(CHANNEL_TAPS - 1):].copy()

        # Interference
        if USE_COMPLEX_IQ:
            interf = INTERF_AMP * np.exp(1j * 2.0 * np.pi * INTERF_FREQ * t)
            interf = interf.astype(np.complex64)
        else:
            interf = INTERF_AMP * np.cos(2.0 * np.pi * INTERF_FREQ * t)
            interf = interf.astype(np.float32).astype(np.complex64)

        # Noise
        if USE_COMPLEX_IQ:
            noise_re = self.rng.randn(n).astype(np.float32)
            noise_im = self.rng.randn(n).astype(np.float32)
            noise = (NOISE_AMP / np.sqrt(2.0)) * (noise_re + 1j * noise_im)
            noise = noise.astype(np.complex64)
        else:
            noise = (NOISE_AMP * self.rng.randn(n)).astype(np.float32).astype(np.complex64)

        d = d + interf + noise

        self.idx += n

        return x.astype(np.complex64), d.astype(np.complex64)


# ============================================================
# 1-tap LMS software core
# ============================================================

class LMS1TapSoftware:
    def __init__(self, mu=0.005):
        self.mu = np.float32(mu)
        self.w = np.complex64(0.0 + 0.0j)

    def process_chunk(self, x, d):
        """
        1-tap LMS:

          y[n] = w * x[n]
          e[n] = d[n] - y[n]
          w    = w + mu * e[n] * conj(x[n])
        """
        N = len(x)

        y = np.zeros(N, dtype=np.complex64)
        e = np.zeros(N, dtype=np.complex64)

        for n in range(N):
            y_n = self.w * x[n]
            e_n = d[n] - y_n

            self.w = self.w + self.mu * e_n * np.conj(x[n])

            y[n] = y_n
            e[n] = e_n

        return y, e, np.complex64(self.w)


# ============================================================
# Theoretical floors
# ============================================================

# Absolute noise/interference floor
noise_floor = float(np.sqrt(INTERF_AMP ** 2 + NOISE_AMP ** 2))

# 1-tap cannot model h1, h2, h3, so these act like residual ISI
isi_power = float(np.sum(np.abs(H_CHANNEL[1:]) ** 2) * PILOT_AMP ** 2)
isi_floor = float(np.sqrt(isi_power + INTERF_AMP ** 2 + NOISE_AMP ** 2))


# ============================================================
# Main test
# ============================================================

print("=" * 72)
print("  1-Tap LMS Software-Only Test")
print("=" * 72)

print("\n[1] Configuration")
print("Chunks        :", NUM_CHUNKS)
print("Chunk size    :", CHUNK_SIZE)
print("Total samples :", NUM_CHUNKS * CHUNK_SIZE)
print("Channel taps  :", H_CHANNEL)
print("LMS taps      :", LMS_TAPS)
print("mu            :", MU_FLOAT)
print("Complex IQ    :", USE_COMPLEX_IQ)
print("Q15 emulation :", EMULATE_Q15_INPUT)
print("Noise floor   : {:.5f}".format(noise_floor))
print("1-tap ISI floor approx: {:.5f}".format(isi_floor))
print()

gen = SignalGenerator(seed=42)
lms = LMS1TapSoftware(mu=MU_FLOAT)

all_err = []
all_snr = []
all_time_ms = []
all_proc_ms = []
all_w = []

last_x = None
last_d = None
last_y = None

total_samples = NUM_CHUNKS * CHUNK_SIZE
t_total_start = time.time()

for ci in range(NUM_CHUNKS):
    t0 = time.perf_counter()

    x, d = gen.next_chunk(CHUNK_SIZE)

    if EMULATE_Q15_INPUT:
        x = quantize_complex_q15(x)
        d = quantize_complex_q15(d)

    y, e, w_now = lms.process_chunk(x, d)

    if EMULATE_Q15_INPUT:
        y = quantize_complex_q15(y)
        e = d - y

    err_rms = float(np.sqrt(np.mean(np.abs(e) ** 2)))
    sig_rms = float(np.sqrt(np.mean(np.abs(d) ** 2)))
    snr_db = 10.0 * np.log10(sig_rms ** 2 / (err_rms ** 2 + 1e-12))

    t_proc = time.perf_counter() - t0

    all_err.append(err_rms)
    all_snr.append(snr_db)
    all_time_ms.append(ci * CHUNK_SIZE / SAMP_RATE * 1e3)
    all_proc_ms.append(t_proc * 1e3)
    all_w.append(w_now)

    last_x = x.copy()
    last_d = d.copy()
    last_y = y.copy()

    if ci % LOG_EVERY == 0 or ci == NUM_CHUNKS - 1:
        elapsed = time.time() - t_total_start
        throughput = ((ci + 1) * CHUNK_SIZE) / elapsed / 1e6

        x_rms = float(np.sqrt(np.mean(np.abs(x) ** 2)))
        d_rms = float(np.sqrt(np.mean(np.abs(d) ** 2)))
        y_rms = float(np.sqrt(np.mean(np.abs(y) ** 2)))

        print(
            "Chunk {:>4}/{} | e_rms={:.5f} | SNR={:+.2f} dB | "
            "x_rms={:.4f} d_rms={:.4f} y_rms={:.4f} | "
            "proc={:.3f} ms | tput={:.3f} MSps | w={:+.4f}{:+.4f}j | {:.2f}s".format(
                ci + 1,
                NUM_CHUNKS,
                err_rms,
                snr_db,
                x_rms,
                d_rms,
                y_rms,
                np.mean(all_proc_ms),
                throughput,
                w_now.real,
                w_now.imag,
                elapsed
            )
        )


# ============================================================
# Results summary
# ============================================================

t_total = time.time() - t_total_start

all_err = np.array(all_err)
all_snr = np.array(all_snr)
all_time_ms = np.array(all_time_ms)
all_proc_ms = np.array(all_proc_ms)
all_w = np.array(all_w)

error_reduction = (1.0 - all_err[-1] / (all_err[0] + 1e-12)) * 100.0
throughput = total_samples / t_total / 1e6

print("\n" + "=" * 72)
print("  SOFTWARE RESULTS — 1-Tap LMS")
print("=" * 72)

print("Total samples          : {:,}".format(total_samples))
print("Total script time      : {:.6f} s".format(t_total))
print("Throughput total       : {:.4f} MSamples/s".format(throughput))

print("\nTiming:")
print("Average processing     : {:.3f} ms/chunk".format(float(np.mean(all_proc_ms))))

print("\nSignal / LMS metrics:")
print("Error RMS start        : {:.5f}".format(float(all_err[0])))
print("Error RMS end          : {:.5f}".format(float(all_err[-1])))
print("Noise floor approx     : {:.5f}".format(noise_floor))
print("1-tap ISI floor approx : {:.5f}".format(isi_floor))
print("Error reduction        : {:.1f}%".format(error_reduction))
print("Final SNR              : {:.2f} dB".format(float(all_snr[-1])))

print("\nWeight:")
print("Target main tap h0     : {}".format(H_CHANNEL[0]))
print("Final learned weight   : {}".format(lms.w))
print("Main tap error mag     : {:.5f}".format(float(np.abs(lms.w - H_CHANNEL[0]))))

print("\nNote:")
print("This is a 1-tap LMS trying to equalize a 4-tap channel.")
print("So it is not expected to reach the noise floor.")
print("It should usually stop closer to the ISI floor than the 4-tap LMS.")


# ============================================================
# Plot
# ============================================================

print("\n[2] Plotting...")

plt.figure(figsize=(13, 10))

plt.subplot(4, 1, 1)
plt.plot(all_time_ms, all_err, label="Software 1-Tap Error RMS")
plt.axhline(noise_floor, linestyle="--", label="Noise floor approx")
plt.axhline(isi_floor, linestyle=":", label="1-tap ISI floor approx")
plt.title("Software 1-Tap LMS — Error RMS")
plt.xlabel("Time (ms)")
plt.ylabel("Error RMS")
plt.grid(True)
plt.legend()

plt.subplot(4, 1, 2)
plt.plot(all_time_ms, all_snr, label="SNR")
plt.title("Software 1-Tap LMS — SNR")
plt.xlabel("Time (ms)")
plt.ylabel("SNR (dB)")
plt.grid(True)
plt.legend()

plt.subplot(4, 1, 3)
plt.plot(all_time_ms, all_w.real, label="w real")
plt.plot(all_time_ms, all_w.imag, label="w imag")
plt.axhline(H_CHANNEL[0].real, linestyle="--", label="target h0 real")
plt.title("Software 1-Tap LMS — Learned Weight")
plt.xlabel("Time (ms)")
plt.ylabel("Weight")
plt.grid(True)
plt.legend()

plt.subplot(4, 1, 4)
nshow = min(256, len(last_d))
t_us = np.arange(nshow) / SAMP_RATE * 1e6

plt.plot(t_us, last_x[:nshow].real, label="x real")
plt.plot(t_us, last_d[:nshow].real, label="d real")
plt.plot(t_us, last_y[:nshow].real, label="y real")
plt.title("Final Chunk Waveform Snapshot")
plt.xlabel("Time (us)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.legend()

plt.tight_layout()

out_png = "software_1tap_lms_results.png"
plt.savefig(out_png, dpi=150)
print("Saved plot to:", out_png)

try:
    from IPython.display import Image, display
    display(Image(out_png))
except Exception:
    pass

print("\nDone.")