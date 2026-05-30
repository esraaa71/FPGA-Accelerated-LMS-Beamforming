# ============================================================
# PYNQ Jupyter Code - 1 Tap LMS, HARDWARE VERSION
# Structure mirrors 1_tap_software.py exactly.
# The only difference: LMS computation is offloaded to the
# FPGA (PL) via AXI DMA instead of being done in Python.
#
# GNU Radio side is used exactly the same way as the software
# version — it sends UDP packets and receives y / |e| back.
#
# Hardware path:
#   Python -> AXI DMA MM2S -> LMS FPGA IP -> AXI DMA S2MM -> Python
#
# Input word (64-bit):  { xr[63:48] | xi[47:32] | dr[31:16] | di[15:0] }
# Output word (64-bit): { yr[63:48] | yi[47:32] | er[31:16] | ei[15:0] }
# All values are Q1.15 signed integers.
# ============================================================

import numpy as np
import socket
import struct
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pynq import Overlay, allocate

# ============================================================
# Bitstream / Overlay
# ============================================================
# NOTE: Place single_tap_iq.bit and single_tap_iq.hwh in the
#       SAME directory as this script before running on PYNQ.
BITSTREAM = "single_tap_iq.bit"

# ============================================================
# UDP settings  (must match GNU Radio flowgraph)
# ============================================================
UDP_IP   = "0.0.0.0"
UDP_PORT = 5005

CHUNK_SIZE    = 1024
NUM_CHUNKS    = 300
TOTAL_SAMPLES = CHUNK_SIZE * NUM_CHUNKS

# ============================================================
# LMS / Signal settings
# ============================================================
MU_FLOAT = 0.005
Q15      = 32768.0

# True  = ignore GNU x,d and generate internal 4-tap test -> expected ~49%-50%
# False = use GNU Radio x0, d0 directly
USE_INTERNAL_4TAP_TEST = True

# Internal signal generation parameters  (same as software reference)
SAMP_RATE   = 100e6
PILOT_FREQS = [1e6, 7e6, 18e6, 35e6]
PILOT_AMP   = 0.30

INTERF_FREQ = 45e6
INTERF_AMP  = 0.02
NOISE_AMP   = 0.005

H_CHANNEL = np.array([0.60, 0.50, -0.40, 0.30], dtype=np.complex64)

# Pipeline latency of the FPGA LMS core (samples to discard at start)
PIPE_LATENCY = 7

LOG_EVERY    = 30
SAVE_PLOT    = True
SAVE_CAPTURE = True


# ============================================================
# Q1.15 helpers  (identical to software version)
# ============================================================

def to_q15(arr):
    """Convert float32 array to signed Q1.15 int16 with saturation."""
    arr = np.asarray(arr, dtype=np.float32)
    arr = np.clip(arr, -1.0, 1.0 - 1.0 / Q15)
    return np.round(arr * Q15).astype(np.int16)


def from_q15(u16_arr):
    """Interpret uint16 bits as signed Q1.15 float32."""
    return u16_arr.astype(np.uint16).view(np.int16).astype(np.float32) / Q15


def quantize_complex_q15(x):
    """Round complex signal to Q1.15 resolution."""
    xr = from_q15(to_q15(x.real).view(np.uint16))
    xi = from_q15(to_q15(x.imag).view(np.uint16))
    return (xr + 1j * xi).astype(np.complex64)


# ============================================================
# AXI-Stream packing helpers
# ============================================================

def pack_words_64(x, d):
    """
    Pack complex x and d into uint64 AXI-Stream words.

    Word layout (matches FPGA RTL):
      tdata[63:48] = x_real  (Q1.15)
      tdata[47:32] = x_imag  (Q1.15)
      tdata[31:16] = d_real  (Q1.15)
      tdata[15: 0] = d_imag  (Q1.15)
    """
    xr = to_q15(x.real).view(np.uint16).astype(np.uint64)
    xi = to_q15(x.imag).view(np.uint16).astype(np.uint64)
    dr = to_q15(d.real).view(np.uint16).astype(np.uint64)
    di = to_q15(d.imag).view(np.uint16).astype(np.uint64)
    return (xr << 48) | (xi << 32) | (dr << 16) | di


def unpack_words_64(out_u64):
    """
    Unpack uint64 AXI-Stream words received from FPGA.

    Word layout (matches FPGA RTL):
      tdata[63:48] = y_real  (Q1.15)
      tdata[47:32] = y_imag  (Q1.15)
      tdata[31:16] = e_real  (Q1.15)
      tdata[15: 0] = e_imag  (Q1.15)

    Returns:
      y  complex64 array
      e  complex64 array
    """
    out_u64 = np.asarray(out_u64, dtype=np.uint64)
    yr_u16  = ((out_u64 >> 48) & 0xFFFF).astype(np.uint16)
    yi_u16  = ((out_u64 >> 32) & 0xFFFF).astype(np.uint16)
    er_u16  = ((out_u64 >> 16) & 0xFFFF).astype(np.uint16)
    ei_u16  = ( out_u64        & 0xFFFF).astype(np.uint16)

    y = (from_q15(yr_u16) + 1j * from_q15(yi_u16)).astype(np.complex64)
    e = (from_q15(er_u16) + 1j * from_q15(ei_u16)).astype(np.complex64)
    return y, e


# ============================================================
# Internal 4-tap signal generator  (identical to software version)
# ============================================================

class Internal4TapSignalGenerator:
    def __init__(self, seed=42):
        self.idx    = 0
        self.rng    = np.random.RandomState(seed)
        self.prev_x = np.zeros(len(H_CHANNEL) - 1, dtype=np.complex64)

    def next_chunk(self, n):
        t = np.arange(self.idx, self.idx + n, dtype=np.float64) / SAMP_RATE

        # Multi-tone pilot / reference
        x = np.zeros(n, dtype=np.complex64)
        for f in PILOT_FREQS:
            tone = np.cos(2.0 * np.pi * f * t).astype(np.float32)
            x   += (PILOT_AMP / len(PILOT_FREQS)) * tone.astype(np.complex64)

        # 4-tap channel output (d)
        x_hist = np.concatenate([self.prev_x, x]).astype(np.complex64)
        conv   = np.convolve(x_hist, H_CHANNEL, mode='full').astype(np.complex64)
        d      = conv[len(H_CHANNEL) - 1 : len(H_CHANNEL) - 1 + n].copy()
        self.prev_x = x_hist[-(len(H_CHANNEL) - 1):].copy()

        # Interference + noise
        interf = INTERF_AMP * np.cos(2.0 * np.pi * INTERF_FREQ * t)
        interf = interf.astype(np.float32).astype(np.complex64)
        noise  = NOISE_AMP * self.rng.randn(n).astype(np.float32).astype(np.complex64)

        d = d + interf + noise
        self.idx += n
        return x.astype(np.complex64), d.astype(np.complex64)


# ============================================================
# FPGA DMA transfer helper
# ============================================================

def dma_transfer(words_in):
    """
    Send packed 64-bit words to FPGA LMS core and return packed output.

    Fires both DMA channels simultaneously and waits for completion.
    This replaces lms.process_chunk() from the software version.
    """
    in_buf[:]  = words_in
    out_buf[:] = 0

    dma.recvchannel.transfer(out_buf)   # arm S2MM first
    dma.sendchannel.transfer(in_buf)    # then fire MM2S

    dma.sendchannel.wait()
    dma.recvchannel.wait()

    return out_buf[:].copy()


# ============================================================
# Banner
# ============================================================

print("=" * 72)
print("PYNQ / GNU 1-Tap LMS Test - HARDWARE VERSION - Expected ~49%-50%")
print("=" * 72)

print("\nConfiguration:")
print("Bitstream             :", BITSTREAM)
print("UDP port              :", UDP_PORT)
print("Chunk size            :", CHUNK_SIZE)
print("Chunks                :", NUM_CHUNKS)
print("Total samples         :", TOTAL_SAMPLES)
print("mu                    :", MU_FLOAT)
print("Pipeline latency      :", PIPE_LATENCY)
print("Use internal 4-tap d  :", USE_INTERNAL_4TAP_TEST)

noise_floor = float(np.sqrt(INTERF_AMP ** 2 + NOISE_AMP ** 2))
isi_power   = float(np.sum(np.abs(H_CHANNEL[1:]) ** 2) * PILOT_AMP ** 2)
isi_floor   = float(np.sqrt(isi_power + INTERF_AMP ** 2 + NOISE_AMP ** 2))

print("Noise floor approx    : {:.5f}".format(noise_floor))
print("1-tap ISI floor approx: {:.5f}".format(isi_floor))


# ============================================================
# Load FPGA overlay
# ============================================================

print("\n[1] Loading FPGA overlay: {} ...".format(BITSTREAM))
try:
    overlay = Overlay(BITSTREAM)
except Exception as e:
    print("ERROR: Failed to load {}.".format(BITSTREAM))
    print("  -> Make sure {0} and {1} are in the same directory as this script.".format(
        BITSTREAM, BITSTREAM.replace('.bit', '.hwh')))
    raise e

print("Overlay loaded.")

print("\nAvailable IPs:")
for ip_name in overlay.ip_dict.keys():
    print("  ", ip_name)

# Fetch DMA and GPIO handles
# NOTE: Names must match the Vivado block design used to generate the .bit file.
dma     = overlay.axi_dma_0
gpio_mu = overlay.axi_gpio_0.channel1

# Write step size (mu) to hardware via AXI GPIO
mu_q15_raw = int(np.round(MU_FLOAT * Q15))
mu_gpio    = mu_q15_raw if mu_q15_raw >= 0 else mu_q15_raw + 65536  # unsigned 16-bit
gpio_mu.write(mu_gpio, 0xFFFF)

print("DMA handle            : axi_dma_0")
print("GPIO mu handle        : axi_gpio_0.channel1")
print("mu written to GPIO    : {} -> Q1.15 {}".format(MU_FLOAT, mu_gpio))


# ============================================================
# Allocate contiguous DMA buffers
# ============================================================

print("\n[2] Allocating DMA buffers ({} x uint64) ...".format(CHUNK_SIZE))
in_buf  = allocate(shape=(CHUNK_SIZE,), dtype=np.uint64)
out_buf = allocate(shape=(CHUNK_SIZE,), dtype=np.uint64)
print("Input buffer  : shape={}, dtype={}".format(in_buf.shape,  in_buf.dtype))
print("Output buffer : shape={}, dtype={}".format(out_buf.shape, out_buf.dtype))


# ============================================================
# UDP socket  (same as software version)
# ============================================================

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

try:
    sock.bind((UDP_IP, UDP_PORT))
except OSError as e:
    print("\nERROR: UDP port is busy:", UDP_PORT)
    print("Run this in terminal if needed:")
    print("sudo fuser -k 5005/udp")
    in_buf.freebuffer()
    out_buf.freebuffer()
    raise e

sock.settimeout(5.0)

print("\n[3] Listening on {}:{} ...".format(UDP_IP, UDP_PORT))
print("Now start GNU Radio.")
print()


# ============================================================
# Main loop  (mirrors software version — only LMS call differs)
# ============================================================

gen = Internal4TapSignalGenerator(seed=42)

chunk_cnt     = 0
sample_cnt    = 0
bad_chunk_cnt = 0

all_err     = []
all_snr     = []
all_hw_ms   = []
all_proc_ms = []

all_x_chunks = []
all_d_chunks = []
all_y_chunks = []
all_e_chunks = []

last_x = None
last_d = None
last_y = None

t_start = None

try:
    while chunk_cnt < NUM_CHUNKS:
        # --------------------------------------------------------
        # Wait for UDP packet from GNU Radio  (same as software)
        # --------------------------------------------------------
        try:
            data, addr = sock.recvfrom(65507)
        except socket.timeout:
            continue

        if len(data) < 12:
            continue

        # Header: uint32 N | uint32 num_ant | float32 mu
        N, num_ant, mu_rx = struct.unpack_from("<IIf", data, 0)

        if N != CHUNK_SIZE:
            bad_chunk_cnt += 1
            if bad_chunk_cnt <= 5:
                print("Bad N from GNU:", N, "Expected:", CHUNK_SIZE)
                print("Set Max_Chunk = 1024 in GNU block.")
            continue

        expected_total = 12 + N * 3 * 8
        if len(data) < expected_total:
            print("Bad packet size. Got {}, expected {}".format(len(data), expected_total))
            continue

        if t_start is None:
            t_start = time.time()
            print("First valid packet from:", addr)
            print("Measurement started.\n")

        t0 = time.perf_counter()

        # --------------------------------------------------------
        # Read GNU Radio payload  (same as software)
        # --------------------------------------------------------
        off = 12
        x0_gnu = np.frombuffer(data, dtype=np.complex64, count=N, offset=off); off += N * 8
        x1_gnu = np.frombuffer(data, dtype=np.complex64, count=N, offset=off); off += N * 8
        d0_gnu = np.frombuffer(data, dtype=np.complex64, count=N, offset=off)

        # --------------------------------------------------------
        # Choose signal source  (same as software)
        # --------------------------------------------------------
        if USE_INTERNAL_4TAP_TEST:
            x, d = gen.next_chunk(N)
        else:
            x = np.array(x0_gnu, dtype=np.complex64, copy=True)
            d = np.array(d0_gnu, dtype=np.complex64, copy=True)

        # Pre-quantise to Q1.15  (matches hardware precision — same as software EMULATE_Q15)
        x = quantize_complex_q15(x)
        d = quantize_complex_q15(d)

        # --------------------------------------------------------
        # FPGA LMS computation via AXI DMA
        # (replaces lms.process_chunk(x, d) from software version)
        # --------------------------------------------------------
        words_in = pack_words_64(x, d)

        t_hw0        = time.perf_counter()
        result_words = dma_transfer(words_in)
        hw_ms        = (time.perf_counter() - t_hw0) * 1000.0

        # --------------------------------------------------------
        # Unpack FPGA output
        # --------------------------------------------------------
        y_hw, e_hw = unpack_words_64(result_words)

        # Discard first PIPE_LATENCY samples (hardware pipeline fill)
        # and pad to keep chunk length uniform — mirrors pipeline delay correction
        if PIPE_LATENCY > 0 and PIPE_LATENCY < N:
            y = y_hw[PIPE_LATENCY:]
            e = e_hw[PIPE_LATENCY:]
            y = np.concatenate([y, np.full(PIPE_LATENCY, y[-1], dtype=np.complex64)])
            e = np.concatenate([e, np.full(PIPE_LATENCY, e[-1], dtype=np.complex64)])
        else:
            y = y_hw
            e = e_hw

        e_mag = np.abs(e).astype(np.float32)

        # --------------------------------------------------------
        # Send reply to GNU Radio: y complex64[N] + |e| float32[N]
        # (same format as software version)
        # --------------------------------------------------------
        reply = y.astype(np.complex64).tobytes() + e_mag.tobytes()
        sock.sendto(reply, addr)

        proc_ms = (time.perf_counter() - t0) * 1000.0

        # --------------------------------------------------------
        # Metrics  (same as software version)
        # --------------------------------------------------------
        chunk_cnt  += 1
        sample_cnt += N

        err_rms = float(np.sqrt(np.mean(e_mag ** 2)))
        x_rms   = float(np.sqrt(np.mean(np.abs(x) ** 2)))
        d_rms   = float(np.sqrt(np.mean(np.abs(d) ** 2)))
        y_rms   = float(np.sqrt(np.mean(np.abs(y) ** 2)))
        snr_db  = 10.0 * np.log10((d_rms ** 2) / (err_rms ** 2 + 1e-12))

        all_err.append(err_rms)
        all_snr.append(snr_db)
        all_hw_ms.append(hw_ms)
        all_proc_ms.append(proc_ms)

        if SAVE_CAPTURE:
            all_x_chunks.append(x.copy())
            all_d_chunks.append(d.copy())
            all_y_chunks.append(y.copy())
            all_e_chunks.append(e.copy())

        last_x = x.copy()
        last_d = d.copy()
        last_y = y.copy()

        elapsed    = time.time() - t_start
        throughput = sample_cnt / elapsed / 1e6 if elapsed > 0 else 0.0

        if chunk_cnt == 1:
            print("First chunk RMS:")
            print("  x_rms = {:.5f}".format(x_rms))
            print("  d_rms = {:.5f}".format(d_rms))
            print()

        if chunk_cnt == 1 or (chunk_cnt - 1) % LOG_EVERY == 0 or chunk_cnt == NUM_CHUNKS:
            current_reduction = (1.0 - all_err[-1] / (all_err[0] + 1e-12)) * 100.0
            print(
                "Chunk {:>4}/{} | e_rms={:.5f} | reduction={:.1f}% | "
                "SNR={:+.2f} dB | x_rms={:.4f} d_rms={:.4f} y_rms={:.4f} | "
                "hw={:.3f} ms | proc={:.3f} ms | tput={:.3f} MSps".format(
                    chunk_cnt,
                    NUM_CHUNKS,
                    err_rms,
                    current_reduction,
                    snr_db,
                    x_rms,
                    d_rms,
                    y_rms,
                    hw_ms,
                    float(np.mean(all_proc_ms)),
                    throughput,
                )
            )

    print("\nReached target:", TOTAL_SAMPLES, "samples")

except KeyboardInterrupt:
    print("\nStopped by user.")

finally:
    sock.close()
    print("\nSocket closed.")

    # Free DMA buffers
    in_buf.freebuffer()
    out_buf.freebuffer()
    print("DMA buffers freed.")

    all_err     = np.array(all_err,     dtype=np.float64)
    all_snr     = np.array(all_snr,     dtype=np.float64)
    all_hw_ms   = np.array(all_hw_ms,   dtype=np.float64)
    all_proc_ms = np.array(all_proc_ms, dtype=np.float64)

    if len(all_err) > 0:
        err_start       = float(all_err[0])
        err_end         = float(all_err[-1])
        error_reduction = (1.0 - err_end / (err_start + 1e-12)) * 100.0
        avg_hw_ms       = float(np.mean(all_hw_ms))
        avg_proc_ms     = float(np.mean(all_proc_ms))

        print("\n" + "=" * 72)
        print("FINAL HARDWARE RESULTS")
        print("=" * 72)

        print("Total samples          : {:,}".format(sample_cnt))
        print("Average HW DMA+LMS     : {:.3f} ms/chunk".format(avg_hw_ms))
        print("Min HW DMA+LMS         : {:.3f} ms/chunk".format(float(np.min(all_hw_ms))))
        print("Max HW DMA+LMS         : {:.3f} ms/chunk".format(float(np.max(all_hw_ms))))
        print("Average total proc     : {:.3f} ms/chunk".format(avg_proc_ms))
        print()
        print("Error RMS start        : {:.5f}".format(err_start))
        print("Error RMS end          : {:.5f}".format(err_end))
        print("Error reduction        : {:.1f}%".format(error_reduction))
        print("Final SNR              : {:.2f} dB".format(float(all_snr[-1])))
        print()
        print("Noise floor approx     : {:.5f}".format(noise_floor))
        print("1-tap ISI floor approx : {:.5f}".format(isi_floor))

        if 45.0 <= error_reduction <= 55.0:
            print("\nOK: Error reduction is in the expected 49%-50% range.")
        elif error_reduction > 90.0:
            print("\nWARNING: Reduction is too high.")
            print("This usually means d is too similar to x, or USE_INTERNAL_4TAP_TEST is False.")
        else:
            print("\nNOTE: Reduction is not near 50%. Check parameters or GNU packet flow.")

        # --------------------------------------------------------
        # Save capture  (same fields as software version)
        # --------------------------------------------------------
        if SAVE_CAPTURE:
            np.savez(
                "pynq_lms_hardware_1tap_capture.npz",
                x            = np.concatenate(all_x_chunks),
                d            = np.concatenate(all_d_chunks),
                y            = np.concatenate(all_y_chunks),
                e            = np.concatenate(all_e_chunks),
                err_rms      = all_err,
                snr_db       = all_snr,
                hw_ms        = all_hw_ms,
                mu           = MU_FLOAT,
                chunk_size   = CHUNK_SIZE,
                pipe_latency = PIPE_LATENCY,
            )
            print("\nSaved capture: pynq_lms_hardware_1tap_capture.npz")

        # --------------------------------------------------------
        # Plot  (same 4 subplots as software, hw timing replaces weight plot)
        # --------------------------------------------------------
        if SAVE_PLOT:
            chunks = np.arange(1, len(all_err) + 1)

            plt.figure(figsize=(13, 12))

            # --- Error RMS ---
            plt.subplot(4, 1, 1)
            plt.plot(chunks, all_err, color='#ffa726', lw=2, label="HW Error RMS")
            plt.axhline(noise_floor, linestyle="--", color='#66bb6a', label="Noise floor")
            plt.axhline(isi_floor,   linestyle=":",  color='#ef5350', label="1-tap ISI floor")
            plt.fill_between(chunks, all_err, noise_floor, alpha=0.12, color='#ffa726')
            plt.title("Hardware 1-Tap LMS - Expected ~50% Error Reduction")
            plt.xlabel("Chunk")
            plt.ylabel("Error RMS")
            plt.grid(True, alpha=0.35)
            plt.legend()

            # --- SNR ---
            plt.subplot(4, 1, 2)
            plt.plot(chunks, all_snr, color='#42a5f5', lw=2, label="SNR dB")
            plt.title("SNR")
            plt.xlabel("Chunk")
            plt.ylabel("SNR (dB)")
            plt.grid(True, alpha=0.35)
            plt.legend()

            # --- HW DMA timing (replaces weight plot from software) ---
            plt.subplot(4, 1, 3)
            plt.plot(chunks, all_hw_ms, color='#ab47bc', lw=1.5, label="DMA + HW LMS (ms)")
            plt.axhline(avg_hw_ms, linestyle="--",
                        color='#ce93d8', label="Mean = {:.3f} ms".format(avg_hw_ms))
            plt.title("Hardware DMA + LMS Processing Time per Chunk")
            plt.xlabel("Chunk")
            plt.ylabel("Time (ms)")
            plt.grid(True, alpha=0.35)
            plt.legend()

            # --- Final chunk snapshot ---
            plt.subplot(4, 1, 4)
            nshow = min(256, len(last_d))
            idx   = np.arange(nshow)
            plt.plot(idx, last_x[:nshow].real, label="x real")
            plt.plot(idx, last_d[:nshow].real, label="d real")
            plt.plot(idx, last_y[:nshow].real, label="y (HW) real")
            plt.title("Final Chunk Snapshot")
            plt.xlabel("Sample")
            plt.ylabel("Amplitude")
            plt.grid(True, alpha=0.35)
            plt.legend()

            plt.tight_layout()
            plt.savefig("pynq_lms_hardware_1tap_results.png", dpi=150)
            plt.show()
            print("Saved plot: pynq_lms_hardware_1tap_results.png")

    else:
        print("No valid chunks received.")
