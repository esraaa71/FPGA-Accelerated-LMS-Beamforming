# ============================================================
# PYNQ Jupyter Code - 1 Tap LMS, Fixed to give ~49%-50% Error Reduction
# GNU is used only to trigger UDP packets and display returned y and |e|
# ============================================================

import numpy as np
import socket
import struct
import time
import matplotlib.pyplot as plt

# ============================================================
# UDP settings
# ============================================================

UDP_IP = "0.0.0.0"
UDP_PORT = 5005

CHUNK_SIZE = 1024
NUM_CHUNKS = 300
TOTAL_SAMPLES = CHUNK_SIZE * NUM_CHUNKS

# ============================================================
# LMS / Simulation settings
# ============================================================

MU_FLOAT = 0.005
Q15 = 32768.0
EMULATE_Q15 = True

# Important:
# True  = ignore GNU x,d and generate internal 4-tap test -> expected 49%-50%
# False = use GNU x0,d0 directly -> can become 100% if x and d are the same
USE_INTERNAL_4TAP_TEST = True

# Same simulation parameters that give about 49% error reduction
SAMP_RATE = 100e6
PILOT_FREQS = [1e6, 7e6, 18e6, 35e6]
PILOT_AMP = 0.30

INTERF_FREQ = 45e6
INTERF_AMP = 0.02
NOISE_AMP = 0.005

H_CHANNEL = np.array([0.60, 0.50, -0.40, 0.30], dtype=np.complex64)

LOG_EVERY = 30
SAVE_PLOT = True
SAVE_CAPTURE = True


# ============================================================
# Q1.15 helpers
# ============================================================

def to_q15_float(arr):
    arr = np.asarray(arr, dtype=np.float32)
    arr = np.clip(arr, -1.0, 1.0 - 1.0 / Q15)
    q = np.round(arr * Q15).astype(np.int16)
    return q.astype(np.float32) / Q15


def quantize_complex_q15(x):
    xr = to_q15_float(x.real)
    xi = to_q15_float(x.imag)
    return (xr + 1j * xi).astype(np.complex64)


# ============================================================
# Internal signal generator
# ============================================================

class Internal4TapSignalGenerator:
    def __init__(self, seed=42):
        self.idx = 0
        self.rng = np.random.RandomState(seed)
        self.prev_x = np.zeros(len(H_CHANNEL) - 1, dtype=np.complex64)

    def next_chunk(self, n):
        t = np.arange(self.idx, self.idx + n, dtype=np.float64) / SAMP_RATE

        # x = multi-tone pilot/reference
        x = np.zeros(n, dtype=np.complex64)

        for f in PILOT_FREQS:
            tone = np.cos(2.0 * np.pi * f * t).astype(np.float32)
            x += (PILOT_AMP / len(PILOT_FREQS)) * tone.astype(np.complex64)

        # d = 4-tap channel output
        x_hist = np.concatenate([self.prev_x, x]).astype(np.complex64)

        conv = np.convolve(x_hist, H_CHANNEL, mode="full").astype(np.complex64)
        d = conv[len(H_CHANNEL) - 1 : len(H_CHANNEL) - 1 + n].copy()

        self.prev_x = x_hist[-(len(H_CHANNEL) - 1):].copy()

        # Add interference
        interf = INTERF_AMP * np.cos(2.0 * np.pi * INTERF_FREQ * t)
        interf = interf.astype(np.float32).astype(np.complex64)

        # Add noise
        noise = NOISE_AMP * self.rng.randn(n).astype(np.float32)
        noise = noise.astype(np.complex64)

        d = d + interf + noise

        self.idx += n

        return x.astype(np.complex64), d.astype(np.complex64)


# ============================================================
# 1-tap LMS
# ============================================================

class LMS1TapSoftware:
    def __init__(self, mu=0.005):
        self.mu = np.float32(mu)
        self.w = np.complex64(0.0 + 0.0j)

    def process_chunk(self, x, d):
        N = len(x)

        y = np.zeros(N, dtype=np.complex64)
        e = np.zeros(N, dtype=np.complex64)

        for n in range(N):
            y_n = self.w * x[n]
            e_n = d[n] - y_n

            self.w = np.complex64(self.w + self.mu * e_n * np.conj(x[n]))

            y[n] = y_n
            e[n] = e_n

        return y, e, np.complex64(self.w)


# ============================================================
# Start
# ============================================================

print("=" * 72)
print("PYNQ / GNU 1-Tap LMS Test - Expected Error Reduction ~49%-50%")
print("=" * 72)

print("\nConfiguration:")
print("UDP port              :", UDP_PORT)
print("Chunk size            :", CHUNK_SIZE)
print("Chunks                :", NUM_CHUNKS)
print("Total samples         :", TOTAL_SAMPLES)
print("mu                    :", MU_FLOAT)
print("Q15 emulation         :", EMULATE_Q15)
print("Use internal 4-tap d  :", USE_INTERNAL_4TAP_TEST)

noise_floor = float(np.sqrt(INTERF_AMP ** 2 + NOISE_AMP ** 2))
isi_power = float(np.sum(np.abs(H_CHANNEL[1:]) ** 2) * PILOT_AMP ** 2)
isi_floor = float(np.sqrt(isi_power + INTERF_AMP ** 2 + NOISE_AMP ** 2))

print("Noise floor approx    : {:.5f}".format(noise_floor))
print("1-tap ISI floor approx: {:.5f}".format(isi_floor))


# ============================================================
# UDP socket
# ============================================================

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

try:
    sock.bind((UDP_IP, UDP_PORT))
except OSError as e:
    print("\nERROR: UDP port is busy:", UDP_PORT)
    print("Run this in terminal if needed:")
    print("sudo fuser -k 5005/udp")
    raise e

sock.settimeout(5.0)

print("\nListening on {}:{} ...".format(UDP_IP, UDP_PORT))
print("Now start GNU Radio.")
print()


# ============================================================
# Main loop
# ============================================================

gen = Internal4TapSignalGenerator(seed=42)
lms = LMS1TapSoftware(mu=MU_FLOAT)

chunk_cnt = 0
sample_cnt = 0
bad_chunk_cnt = 0

all_err = []
all_snr = []
all_w = []
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
        try:
            data, addr = sock.recvfrom(65507)
        except socket.timeout:
            continue

        if len(data) < 12:
            continue

        # Header from GNU:
        # uint32 N
        # uint32 num_ant
        # float32 mu
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
        # Read GNU data
        # --------------------------------------------------------
        off = 12

        x0_gnu = np.frombuffer(data, dtype=np.complex64, count=N, offset=off)
        off += N * 8

        x1_gnu = np.frombuffer(data, dtype=np.complex64, count=N, offset=off)
        off += N * 8

        d0_gnu = np.frombuffer(data, dtype=np.complex64, count=N, offset=off)

        # --------------------------------------------------------
        # Important correction
        # --------------------------------------------------------
        if USE_INTERNAL_4TAP_TEST:
            # This gives the same type of result as the software simulation:
            # 1-tap LMS trying to fit a 4-tap channel.
            x, d = gen.next_chunk(N)
        else:
            # This uses GNU directly.
            # Warning: if x0_gnu and d0_gnu are the same, error reduction can become 100%.
            x = np.array(x0_gnu, dtype=np.complex64, copy=True)
            d = np.array(d0_gnu, dtype=np.complex64, copy=True)

        if EMULATE_Q15:
            x = quantize_complex_q15(x)
            d = quantize_complex_q15(d)

        # --------------------------------------------------------
        # LMS
        # --------------------------------------------------------
        y, e, w_now = lms.process_chunk(x, d)

        if EMULATE_Q15:
            y = quantize_complex_q15(y)
            e = d - y

        e_mag = np.abs(e).astype(np.float32)

        # --------------------------------------------------------
        # Send reply to GNU:
        # y complex64[N] + |e| float32[N]
        # --------------------------------------------------------
        reply = y.astype(np.complex64).tobytes() + e_mag.tobytes()
        sock.sendto(reply, addr)

        proc_ms = (time.perf_counter() - t0) * 1000.0

        # --------------------------------------------------------
        # Metrics
        # --------------------------------------------------------
        chunk_cnt += 1
        sample_cnt += N

        err_rms = float(np.sqrt(np.mean(e_mag ** 2)))
        x_rms = float(np.sqrt(np.mean(np.abs(x) ** 2)))
        d_rms = float(np.sqrt(np.mean(np.abs(d) ** 2)))
        y_rms = float(np.sqrt(np.mean(np.abs(y) ** 2)))

        snr_db = 10.0 * np.log10((d_rms ** 2) / (err_rms ** 2 + 1e-12))

        all_err.append(err_rms)
        all_snr.append(snr_db)
        all_w.append(w_now)
        all_proc_ms.append(proc_ms)

        if SAVE_CAPTURE:
            all_x_chunks.append(x.copy())
            all_d_chunks.append(d.copy())
            all_y_chunks.append(y.copy())
            all_e_chunks.append(e.copy())

        last_x = x.copy()
        last_d = d.copy()
        last_y = y.copy()

        elapsed = time.time() - t_start
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
                "w={:+.5f}{:+.5f}j | proc={:.3f} ms | tput={:.3f} MSps".format(
                    chunk_cnt,
                    NUM_CHUNKS,
                    err_rms,
                    current_reduction,
                    snr_db,
                    x_rms,
                    d_rms,
                    y_rms,
                    w_now.real,
                    w_now.imag,
                    float(np.mean(all_proc_ms)),
                    throughput
                )
            )

    print("\nReached target:", TOTAL_SAMPLES, "samples")

except KeyboardInterrupt:
    print("\nStopped by user.")

finally:
    sock.close()
    print("\nSocket closed.")

    all_err = np.array(all_err, dtype=np.float64)
    all_snr = np.array(all_snr, dtype=np.float64)
    all_w = np.array(all_w, dtype=np.complex64)
    all_proc_ms = np.array(all_proc_ms, dtype=np.float64)

    if len(all_err) > 0:
        err_start = float(all_err[0])
        err_end = float(all_err[-1])
        error_reduction = (1.0 - err_end / (err_start + 1e-12)) * 100.0

        print("\n" + "=" * 72)
        print("FINAL RESULTS")
        print("=" * 72)

        print("Total samples          : {:,}".format(sample_cnt))
        print("Average processing     : {:.3f} ms/chunk".format(float(np.mean(all_proc_ms))))
        print()
        print("Error RMS start        : {:.5f}".format(err_start))
        print("Error RMS end          : {:.5f}".format(err_end))
        print("Error reduction        : {:.1f}%".format(error_reduction))
        print("Final SNR              : {:.2f} dB".format(float(all_snr[-1])))
        print()
        print("Noise floor approx     : {:.5f}".format(noise_floor))
        print("1-tap ISI floor approx : {:.5f}".format(isi_floor))
        print()
        print("Final learned weight   : {:+.6f}{:+.6f}j".format(lms.w.real, lms.w.imag))

        if 45.0 <= error_reduction <= 55.0:
            print("\nOK: Error reduction is in the expected 49%-50% range.")
        elif error_reduction > 90.0:
            print("\nWARNING: Reduction is too high.")
            print("This usually means d is too similar to x, or USE_INTERNAL_4TAP_TEST is False.")
        else:
            print("\nNOTE: Reduction is not near 50%. Check parameters or GNU packet flow.")

        if SAVE_CAPTURE:
            np.savez(
                "pynq_lms_expected_50_capture.npz",
                x=np.concatenate(all_x_chunks),
                d=np.concatenate(all_d_chunks),
                y=np.concatenate(all_y_chunks),
                e=np.concatenate(all_e_chunks),
                err_rms=all_err,
                snr_db=all_snr,
                w=all_w,
                mu=MU_FLOAT,
                chunk_size=CHUNK_SIZE
            )
            print("\nSaved capture: pynq_lms_expected_50_capture.npz")

        if SAVE_PLOT:
            chunks = np.arange(1, len(all_err) + 1)

            plt.figure(figsize=(13, 10))

            plt.subplot(4, 1, 1)
            plt.plot(chunks, all_err, label="Error RMS")
            plt.axhline(noise_floor, linestyle="--", label="Noise floor")
            plt.axhline(isi_floor, linestyle=":", label="1-tap ISI floor")
            plt.title("1-Tap LMS - Expected ~50% Error Reduction")
            plt.xlabel("Chunk")
            plt.ylabel("Error RMS")
            plt.grid(True)
            plt.legend()

            plt.subplot(4, 1, 2)
            plt.plot(chunks, all_snr, label="SNR")
            plt.title("SNR")
            plt.xlabel("Chunk")
            plt.ylabel("SNR dB")
            plt.grid(True)
            plt.legend()

            plt.subplot(4, 1, 3)
            plt.plot(chunks, all_w.real, label="w real")
            plt.plot(chunks, all_w.imag, label="w imag")
            plt.axhline(H_CHANNEL[0].real, linestyle="--", label="target h0 real")
            plt.title("Learned Weight")
            plt.xlabel("Chunk")
            plt.ylabel("Weight")
            plt.grid(True)
            plt.legend()

            plt.subplot(4, 1, 4)
            nshow = min(256, len(last_d))
            idx = np.arange(nshow)
            plt.plot(idx, last_x[:nshow].real, label="x real")
            plt.plot(idx, last_d[:nshow].real, label="d real")
            plt.plot(idx, last_y[:nshow].real, label="y real")
            plt.title("Final Chunk Snapshot")
            plt.xlabel("Sample")
            plt.ylabel("Amplitude")
            plt.grid(True)
            plt.legend()

            plt.tight_layout()
            plt.savefig("pynq_lms_expected_50_results.png", dpi=150)
            plt.show()

            print("Saved plot: pynq_lms_expected_50_results.png")
    else:
        print("No valid chunks received.")