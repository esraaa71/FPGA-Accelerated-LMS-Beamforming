When evaluating hardware acceleration architectures on Zynq SoCs, particularly for complex signal processing pipelines like beamforming, analyzing the performance delta between the ARM processing system and the FPGA fabric reveals the exact efficiency gains.

***

# LMS Algorithm Implementation Comparison: Zynq SoC (ARM vs. FPGA)

This document provides a comparative analysis of a 1-Tap and 4-Tap LMS algorithm running as software on the ARM processing system versus hardware-accelerated DMA streaming on the FPGA fabric.

## Performance Metrics Summary

The table below aggregates the software and hardware execution results extracted from the system logs. 

| Metric | 1-Tap Software (ARM) | 1-Tap Hardware (FPGA) | 4-Tap Software (ARM) | 4-Tap Hardware (FPGA) |
| :--- | :--- | :--- | :--- | :--- |
| **Total Samples** | 307,200 | 307,200 | 307,200 | 307,200 |
| **Total Execution Time** | 35.16 s | 3.76 s | 57.43 s | 4.89 s |
| **System Throughput** | 0.0087 MSamples/s | 0.0816 MSamples/s | 0.0053 MSamples/s | 0.0628 MSamples/s |
| **Avg. Processing Time** | 116.98 ms / chunk | 12.20 ms / chunk | 191.12 ms / chunk | 1.63 ms / chunk |
| **Error Reduction** | 49.1% | 48.9% | 84.6% | 84.3% |
| **Final SNR** | 6.04 dB | 6.05 dB | 16.60 dB | 17.21 dB |

---

## Key Analytical Observations

### 1. Hardware Acceleration Gains (Timing & Throughput)
Moving the LMS algorithm from the ARM processor to the FPGA fabric yields a massive performance boost across both filter sizes. 
* **1-Tap Acceleration:** Total script time drops from ~35.16 seconds in software to ~3.76 seconds in hardware. The average processing time per chunk improves from 116.98 ms down to just 12.20 ms (with the actual DMA+FPGA processing taking a mere 0.674 ms of that chunk time). System throughput increases nearly 10x (0.0087 MSps to 0.0816 MSps).
* **4-Tap Acceleration:** The execution time drops from ~57.43 seconds in software down to ~4.89 seconds in hardware. The average processing time per chunk plummets from 191.12 ms to 1.63 ms. The FPGA achieves a throughput of 0.0628 MSps, which is an 11.8x speedup compared to the software's 0.0053 MSps.

### 2. Algorithmic Fidelity (Software vs. Hardware)
The migration to FPGA does not sacrifice mathematical accuracy. Comparing the Signal/LMS metrics reveals that the hardware implementations track almost identically to the software baseline:
* **4-Tap Comparison:** Both environments achieve roughly 84% error reduction, with the hardware yielding a slightly better final SNR (17.21 dB vs 16.60 dB).
* **1-Tap Comparison:** Both the ARM and FPGA implementations sit right at a ~49% error reduction (from ~0.100 RMS start to ~0.051 RMS end) and a final SNR of ~6.04 dB.

### 3. Algorithmic Limitations (1-Tap vs. 4-Tap)
The data clearly demonstrates the physical limits of the filter size relative to the channel. 
* Because the 1-Tap LMS is being tested against a 4-tap multipath channel, it fundamentally lacks the degrees of freedom to model the delayed taps. 
* As noted in the system logs, it hits the 1-Tap ISI (Intersymbol Interference) floor at 0.21313. 
* Consequently, the 1-tap algorithm stops converging far above the actual noise floor (~0.020), topping out at ~6 dB SNR. Expanding to the 4-tap architecture successfully mitigates the ISI, allowing the error to drop down near the noise floor and tripling the final SNR.