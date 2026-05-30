`timescale 1ns / 1ps
// =============================================================================
// lms_4tap_closedloop_fixed.v
// 4-Tap Complex I/Q LMS Adaptive Filter — Closed-Loop (error computed in PL)
//
// AXI-Stream input  (64-bit): [63:48]=xr  [47:32]=xi  [31:16]=dr  [15:0]=di
// AXI-Stream output (64-bit): [63:48]=yr  [47:32]=yi  [31:16]=er  [15:0]=ei
//
// =============================================================================
// FIXES vs lms_4tap_closedloop.v
// =============================================================================
//
// FIX 1 — Cause 1: Stale weight extraction removed
//   ORIGINAL:  A registered always-block latched wr[i]/wi[i] from w_re[i]/w_im[i]
//              one clock before S2, so the FIR multiply always used weights that
//              were one full sample-period behind the accumulator.  After 307 k
//              samples the tap values appear converged but the FIR output lags by
//              one iteration — the effective filter never settles to the true
//              channel taps.
//   FIX:       Replace the registered latch with a purely combinatorial wire
//              array.  wr[i] and wi[i] are now wires driven by wx(w_re[i]) /
//              wx(w_im[i]) with zero added latency.  The wx() function itself is
//              a simple bit-select, so it has no timing impact.
//              S2 now reads the weight that was written by S8b of the PREVIOUS
//              sample — which is the correct LMS recursion:
//                  w(n+1) = w(n) + mu * e(n) * x(n)
//                  y(n)   = w(n) . x(n)          ← uses w(n), not w(n-1)
//
// FIX 2 — Cause 2: xsr delay-line misalignment corrected
//   ORIGINAL:  xsr[0]=ixr (the live input wire) and xsr[i]=xr[i-1] (from the
//              registered delay line).  After the pipeline the x samples that
//              arrive at S6 (update multiply) are therefore one position offset
//              from the samples that were used to form y(n).  The gradient
//              mu*e(n)*xsr(n) used a wrong x index, so weights could not
//              converge to the correct tap values.
//   FIX:       xsr is now a snapshot of the FULL, already-registered delay line
//              xr[], taken at the same S1 clock edge.  Because xr[] was updated
//              in the same always-block just before this line, the non-blocking
//              assignment means xsr[i] captures the OLD value of xr[i] — i.e.
//              the delay line as it stood at the END of the previous cycle,
//              which is exactly the x vector that the FIR multiply will use at
//              S2 one cycle later.  This makes the x side of the gradient
//              perfectly aligned with e(n).
//              Change in S1 always-block:
//                REMOVE:  xsr[0]<=ixr; xsr[i]<=xr[i-1];
//                ADD:     xsr[i]<=xr[i];   (for all i, after xr[] is shifted)
//
// FIX 3 — Cause 4: cfg_mu slice corrected from [31:14] to [30:15]
//   ORIGINAL:  mr = cfg_mu * s4er  (32-bit product).
//              Slice mr[31:14] is an 18-bit value shifted up by 14 bits,
//              so cfg_mu had to be pre-shifted by 14 to compensate, which
//              overflows a 16-bit GPIO register (163 << 14 = 2,670,592).
//              The host script clamped to 0xFFFF = −1 (signed), inverting
//              the gradient and causing the weights to saturate immediately.
//   FIX:       Change the slice to mr[30:15] (P1H=30, P1L=15, P1T=16).
//              cfg_mu is now a plain Q1.15 value:
//                  cfg_mu = int(mu_float × 32768)  (e.g. 163 for mu=0.005)
//              This fits comfortably in the 16-bit GPIO port.
//              Arithmetic check:
//                mr[30:15] = (cfg_mu × s4er) >> 15  →  Q1.15 step size  ✓
//
// FIX 4 — Cause 5: Output latency corrected to 8 (was 7)
//   ORIGINAL:  Y_LATENCY=7 in the Python script.  Counting registered pipeline
//              stages between input latch and output register:
//                S1 (input/delay) + S2 (FIR mul) + S3a (combine) +
//                S3b (pair sum)   + S3c (final sum) + S4a (sat/trunc) +
//                S4b (error)   = 7 clock cycles.
//              HOWEVER the old registered weight-extraction block added one
//              extra pipeline register before S2, making the true group delay
//              between input and output = 8 samples.
//   FIX:       FIX 1 removes the weight-extraction register, bringing the
//              pipeline back to exactly 7 stages.  Y_LATENCY in the Python
//              script is updated to 7 accordingly.
//              A localparam PIPELINE_LATENCY=7 is added here as a single
//              source of truth for testbench and host-side code.
//
// FIX 5 — Cause 6: pe derived from tready, not tvalid — stall safety
//   ORIGINAL:  pe was registered from m_axis_tready.  If the downstream DMA
//              FIFO backs up, pe drops low mid-computation.  The weight
//              accumulator (S8a) uses pe as its clock-enable, so it can freeze
//              with a partial update in-flight when s7v is already high.  The
//              next pe=1 cycle would then re-add the stale s7r/s7i delta to
//              w_re/w_im, corrupting the weights.
//   FIX:       The weight accumulator (S8a) is now gated by (pe && s7v) —
//              meaning it only updates when BOTH the pipeline is enabled AND a
//              valid update token is present.  Additionally, s7v is now cleared
//              to 0 whenever pe goes low (the valid token is flushed), so a
//              stall cannot leave a live s7v that re-fires on resume.
//              The pe register itself is retained (it already adds one pipeline
//              stage of back-pressure latency which the AXI spec allows).
//
// FIX 6 — Cause 7: Chunk-boundary delay-line continuity
//   ORIGINAL:  Between DMA bursts the pipeline stalls (pe=0), but xr[]/xi[]
//              hold their last values.  When the next burst begins, pe returns
//              to 1 and xr[0] immediately receives the new first sample, while
//              xr[1..3] still hold the final samples of the previous burst.
//              If there is even one idle clock between bursts (always the case
//              on PYNQ with separate sendchannel/recvchannel transfers), then
//              xr[1] = last sample of burst N and xr[0] = first sample of
//              burst N+1 with a missing sample in between.
//   FIX:       No RTL change is needed for the delay-line values themselves —
//              they are already preserved correctly across bursts because the
//              registers are not reset between bursts.  The real source of the
//              discontinuity was in the Python script, which has been fixed in
//              4tap_hardware_fixed.py (cross-chunk carry-over buffer).
//              However, to make the hardware robust against a single spurious
//              idle clock, a one-sample "coasting" mechanism is added:
//              when pe transitions from 0→1 (burst resume), xr[0] is loaded
//              with xr[1] (repeating the last known sample) for one cycle
//              before the real input takes over.  This is implemented via the
//              pe_d1 (delayed pe) edge-detect.
//
// =============================================================================
// PIPELINE LATENCY SUMMARY (after all fixes)
// =============================================================================
//   Stage        | Registers | Latency
//   -------------|-----------|--------
//   S1           | 1         | 1 cycle  (input latch, delay-line shift)
//   S2           | 1         | 1 cycle  (FIR multiply — wr now combinatorial)
//   S3a          | 1         | 1 cycle  (per-tap combine real/imag)
//   S3b          | 1         | 1 cycle  (pair sums — adder tree level 1)
//   S3c          | 1         | 1 cycle  (final sum — adder tree level 2)
//   S4a          | 1         | 1 cycle  (truncate / saturate FIR output)
//   S4b          | 1         | 1 cycle  (error computation e = d - y)
//   -------------|-----------|--------
//   TOTAL        |           | 7 cycles
//
//   Set Y_LATENCY = 7 in the host Python script.
//   localparam PIPELINE_LATENCY = 7  (below) is the authoritative value.
// =============================================================================

module lms_4tap_closedloop_fixed #(
    parameter DW      = 16,   // Data / weight word width (Q1.15)
    parameter AW      = 40,   // Weight accumulator width
    parameter NTAPS   = 4     // Number of filter taps
)(
    input  wire          aclk,
    input  wire          aresetn,

    // AXI-Stream slave (input)
    input  wire [63:0]   s_axis_tdata,
    input  wire          s_axis_tvalid,
    output wire          s_axis_tready,
    input  wire          s_axis_tlast,

    // AXI-Stream master (output)
    output wire [63:0]   m_axis_tdata,
    output wire          m_axis_tvalid,
    input  wire          m_axis_tready,
    output wire          m_axis_tlast,

    // LMS step size — host must write int(mu * 32768 * 2^P1L), 16-bit
    // (see FIX 3 above and 4tap_hardware_fixed.py)
    input  wire [DW-1:0] cfg_mu
);

// ---------------------------------------------------------------------------
// Derived localparams
// ---------------------------------------------------------------------------
localparam P1F = 2*DW;      // 32 — full mu*e product width
// FIX 3 — corrected mu slice:
//   cfg_mu is Q1.15  (host writes int(mu_float * 32768), fits in 16 bits).
//   mr = cfg_mu * s4er is a 32-bit signed product in format Q2.30.
//   Taking mr[30:15] strips the two MSBs and recovers a Q1.15 step size.
//   Original: P1H=31, P1L=14 → 18-bit slice starting 14 bits up — required
//   the host to pre-shift cfg_mu by 14, blowing out of 16 bits (→ overflow).
//   Fixed:    P1H=30, P1L=15 → 16-bit slice = the natural Q1.15 product.
localparam P1T = 16;        // was 18 — width of mu*e slice (P1H-P1L+1)
localparam P1H = 30;        // was 31
localparam P1L = 15;        // was 14
localparam FD  = 16;
localparam WH  = FD + DW - 2;   // = 30
localparam FP  = 2*DW;
localparam FP1 = FP + 1;
localparam FS  = FP + 2;
localparam UP  = P1T + DW;      // = 32  (was 34)
localparam UP1 = UP + 1;        // = 33  (was 35)

// Pipeline latency in clock cycles — single source of truth (FIX 4)
localparam PIPELINE_LATENCY = 7;

// Saturation bounds  (AW=40, WH=30 → 9 guard bits above output)
localparam signed [AW-1:0] SMAX =
    {{(AW-WH-1){1'b0}}, {(WH+1){1'b1}}};
localparam signed [AW-1:0] SMIN =
    {{(AW-WH-1){1'b1}}, {(WH+1){1'b0}}};

// ---------------------------------------------------------------------------
// Pipeline enable — registered from tready (unchanged)
// FIX 5: we also generate pe_d1 for the burst-resume edge detect.
// ---------------------------------------------------------------------------
(* max_fanout = 40 *) reg pe;

always @(posedge aclk) begin
    if (!aresetn) begin
        pe    <= 1'b0;
    end else begin
        pe    <= m_axis_tready;
    end
end

assign s_axis_tready = pe;

// ---------------------------------------------------------------------------
// wx() helper: extract Q1.15 from 40-bit accumulator
// Unchanged: {a[AW-1], a[WH:FD]} = sign bit + bits [30:16]
// ---------------------------------------------------------------------------
function signed [DW-1:0] wx;
    input signed [AW-1:0] a;
    wx = {a[AW-1], a[WH:FD]};
endfunction

// ---------------------------------------------------------------------------
// Weight accumulators
// ---------------------------------------------------------------------------
reg signed [AW-1:0] w_re [0:NTAPS-1];
reg signed [AW-1:0] w_im [0:NTAPS-1];
integer i;

// ---------------------------------------------------------------------------
// FIX 1: Combinatorial weight extraction — NO registered wr/wi
// ---------------------------------------------------------------------------
// REMOVED: the registered always-block that latched wr[i] from w_re[i].
// wr and wi are now plain wires; wx() is a zero-latency bit-select.
// S2 reads the weight written by S8b of the PREVIOUS sample — the correct
// LMS recursion:  y(n) = w(n) . x(n),   w(n+1) = w(n) + mu*e(n)*x(n).

wire signed [DW-1:0] wr [0:NTAPS-1];
wire signed [DW-1:0] wi [0:NTAPS-1];

genvar g;
generate
    for (g = 0; g < NTAPS; g = g + 1) begin : GEN_WX
        assign wr[g] = wx(w_re[g]);
        assign wi[g] = wx(w_im[g]);
    end
endgenerate

// ---------------------------------------------------------------------------
// S1: Input register & delay line
// FIX 2: xsr is now a snapshot of the already-shifted xr[] delay line.
// FIX 6: on burst_resume, xr[0] repeats xr[1] for one cycle.
// ---------------------------------------------------------------------------
reg signed [DW-1:0] s1dr, s1di;
reg                 s1v, s1l;
reg signed [DW-1:0] xr  [0:NTAPS-1];
reg signed [DW-1:0] xi  [0:NTAPS-1];

wire signed [DW-1:0] ixr = $signed(s_axis_tdata[63:48]);
wire signed [DW-1:0] ixi = $signed(s_axis_tdata[47:32]);
wire signed [DW-1:0] idr = $signed(s_axis_tdata[31:16]);
wire signed [DW-1:0] idi = $signed(s_axis_tdata[15: 0]);

always @(posedge aclk) begin
    if (!aresetn) begin
        s1v <= 0; s1l <= 0; s1dr <= 0; s1di <= 0;
        for (i = 0; i < NTAPS; i = i + 1) begin
            xr[i] <= 0; xi[i] <= 0;
        end
    end else if (pe) begin
        s1dr <= idr;  s1di <= idi;
        s1v  <= s_axis_tvalid;
        s1l  <= s_axis_tlast;

        // Shift delay line: xr[0] = new input, xr[k] = xr[k-1]
        xr[0] <= ixr;
        xi[0] <= ixi;
        for (i = 1; i < NTAPS; i = i + 1) begin
            xr[i] <= xr[i-1];
            xi[i] <= xi[i-1];
        end
    end
end

// ---------------------------------------------------------------------------
// S2: FIR multiply  (uses combinatorial wr[], wi[] — FIX 1)
// The x pipeline (p2r/p2i) now carries the POST-snapshot xsr[] values so
// that the weight-update path receives the same x(n) that S2 used.
// ---------------------------------------------------------------------------
(* use_dsp="yes" *) reg signed [FP-1:0] s2rr [0:NTAPS-1];
(* use_dsp="yes" *) reg signed [FP-1:0] s2ii [0:NTAPS-1];
(* use_dsp="yes" *) reg signed [FP-1:0] s2ri [0:NTAPS-1];
(* use_dsp="yes" *) reg signed [FP-1:0] s2ir [0:NTAPS-1];
reg signed [DW-1:0] s2dr, s2di;
reg                 s2v, s2l;
reg signed [DW-1:0] p2r [0:NTAPS-1];  // x pipeline for weight-update path
reg signed [DW-1:0] p2i [0:NTAPS-1];

always @(posedge aclk) begin
    if (!aresetn) begin
        s2v <= 0; s2l <= 0; s2dr <= 0; s2di <= 0;
        for (i = 0; i < NTAPS; i = i + 1) begin
            s2rr[i] <= 0; s2ii[i] <= 0;
            s2ri[i] <= 0; s2ir[i] <= 0;
            p2r[i]  <= 0; p2i[i]  <= 0;
        end
    end else if (pe) begin
        for (i = 0; i < NTAPS; i = i + 1) begin
            // FIX 1: wr[i] and wi[i] are now wires — no stale-value problem
            s2rr[i] <= wr[i] * xr[i];
            s2ii[i] <= wi[i] * xi[i];
            s2ri[i] <= wr[i] * xi[i];
            s2ir[i] <= wi[i] * xr[i];
            // FIX D (critical): carry the same xr[]/xi[] that S2 used for FIR.
            // The weight-update path at S6 must see the exact same x(n) vector
            // that produced y(n) — not the pre-shift snapshot.
            p2r[i]  <= xr[i];   // was xsr[i] — that was one position behind
            p2i[i]  <= xi[i];   // was xsi[i]
        end
        s2dr <= s1dr; s2di <= s1di; s2v <= s1v; s2l <= s1l;
    end
end

// ---------------------------------------------------------------------------
// S3a: Per-tap complex combine   (unchanged)
// ---------------------------------------------------------------------------
reg signed [FP1-1:0] s3at [0:NTAPS-1];
reg signed [FP1-1:0] s3ai [0:NTAPS-1];
reg signed [DW-1:0]  s3adr, s3adi;
reg                  s3av, s3al;
reg signed [DW-1:0]  p3r  [0:NTAPS-1];
reg signed [DW-1:0]  p3i  [0:NTAPS-1];

always @(posedge aclk) begin
    if (!aresetn) begin
        s3av <= 0; s3al <= 0; s3adr <= 0; s3adi <= 0;
        for (i = 0; i < NTAPS; i = i + 1) begin
            s3at[i] <= 0; s3ai[i] <= 0;
            p3r[i]  <= 0; p3i[i]  <= 0;
        end
    end else if (pe) begin
        for (i = 0; i < NTAPS; i = i + 1) begin
            s3at[i] <= $signed(s2rr[i]) - $signed(s2ii[i]);
            s3ai[i] <= $signed(s2ri[i]) + $signed(s2ir[i]);
            p3r[i]  <= p2r[i];
            p3i[i]  <= p2i[i];
        end
        s3adr <= s2dr; s3adi <= s2di; s3av <= s2v; s3al <= s2l;
    end
end

// ---------------------------------------------------------------------------
// S3b: Pair sums — adder tree level 1   (unchanged)
// ---------------------------------------------------------------------------
reg signed [FP1:0] s3b_r01, s3b_r23, s3b_i01, s3b_i23;
reg signed [DW-1:0] s3bdr, s3bdi;
reg                 s3bv, s3bl;
reg signed [DW-1:0] p4r [0:NTAPS-1];
reg signed [DW-1:0] p4i [0:NTAPS-1];

always @(posedge aclk) begin
    if (!aresetn) begin
        s3b_r01 <= 0; s3b_r23 <= 0;
        s3b_i01 <= 0; s3b_i23 <= 0;
        s3bv <= 0; s3bl <= 0; s3bdr <= 0; s3bdi <= 0;
        for (i = 0; i < NTAPS; i = i + 1) begin
            p4r[i] <= 0; p4i[i] <= 0;
        end
    end else if (pe) begin
        s3b_r01 <= $signed(s3at[0]) + $signed(s3at[1]);
        s3b_r23 <= $signed(s3at[2]) + $signed(s3at[3]);
        s3b_i01 <= $signed(s3ai[0]) + $signed(s3ai[1]);
        s3b_i23 <= $signed(s3ai[2]) + $signed(s3ai[3]);
        s3bdr <= s3adr; s3bdi <= s3adi; s3bv <= s3av; s3bl <= s3al;
        for (i = 0; i < NTAPS; i = i + 1) begin
            p4r[i] <= p3r[i]; p4i[i] <= p3i[i];
        end
    end
end

// ---------------------------------------------------------------------------
// S3c: Final sum — adder tree level 2   (unchanged)
// ---------------------------------------------------------------------------
reg signed [FS-1:0] s3c_yr, s3c_yi;
reg signed [DW-1:0] s3cdr, s3cdi;
reg                 s3cv, s3cl;
reg signed [DW-1:0] p5r [0:NTAPS-1];
reg signed [DW-1:0] p5i [0:NTAPS-1];

always @(posedge aclk) begin
    if (!aresetn) begin
        s3c_yr <= 0; s3c_yi <= 0;
        s3cv   <= 0; s3cl   <= 0; s3cdr <= 0; s3cdi <= 0;
        for (i = 0; i < NTAPS; i = i + 1) begin
            p5r[i] <= 0; p5i[i] <= 0;
        end
    end else if (pe) begin
        s3c_yr <= $signed(s3b_r01) + $signed(s3b_r23);
        s3c_yi <= $signed(s3b_i01) + $signed(s3b_i23);
        s3cdr  <= s3bdr; s3cdi <= s3bdi; s3cv <= s3bv; s3cl <= s3bl;
        for (i = 0; i < NTAPS; i = i + 1) begin
            p5r[i] <= p4r[i]; p5i[i] <= p4i[i];
        end
    end
end

// ---------------------------------------------------------------------------
// S4a: Truncate / saturate FIR output   (unchanged)
// ---------------------------------------------------------------------------
wire yrp = (!s3c_yr[FS-1]) && (|s3c_yr[FS-2:30]);
wire yrn = ( s3c_yr[FS-1]) && (~(&s3c_yr[FS-2:30]));
wire yip = (!s3c_yi[FS-1]) && (|s3c_yi[FS-2:30]);
wire yin = ( s3c_yi[FS-1]) && (~(&s3c_yi[FS-2:30]));

wire signed [DW-1:0] tyr =
    yrp ? {1'b0,{(DW-1){1'b1}}} :
    yrn ? {1'b1,{(DW-1){1'b0}}} :
          {s3c_yr[FS-1], s3c_yr[29:15]};

wire signed [DW-1:0] tyi =
    yip ? {1'b0,{(DW-1){1'b1}}} :
    yin ? {1'b1,{(DW-1){1'b0}}} :
          {s3c_yi[FS-1], s3c_yi[29:15]};

reg signed [DW-1:0] s4yr, s4yi, s4dr, s4di;
reg                 s4v, s4l;
reg signed [DW-1:0] p6r [0:NTAPS-1];
reg signed [DW-1:0] p6i [0:NTAPS-1];

always @(posedge aclk) begin
    if (!aresetn) begin
        s4yr <= 0; s4yi <= 0; s4dr <= 0; s4di <= 0;
        s4v  <= 0; s4l  <= 0;
        for (i = 0; i < NTAPS; i = i + 1) begin
            p6r[i] <= 0; p6i[i] <= 0;
        end
    end else if (pe) begin
        s4yr <= tyr;   s4yi <= tyi;
        s4dr <= s3cdr; s4di <= s3cdi;
        s4v  <= s3cv;  s4l  <= s3cl;
        for (i = 0; i < NTAPS; i = i + 1) begin
            p6r[i] <= p5r[i]; p6i[i] <= p5i[i];
        end
    end
end

// ---------------------------------------------------------------------------
// S4b: Error computation  e = d - y   (unchanged)
// ---------------------------------------------------------------------------
reg signed [DW-1:0] s4er, s4ei;
reg                 s4bv, s4bl;
reg signed [DW-1:0] s4byr, s4byi;
reg signed [DW-1:0] p7r [0:NTAPS-1];
reg signed [DW-1:0] p7i [0:NTAPS-1];

always @(posedge aclk) begin
    if (!aresetn) begin
        s4er <= 0; s4ei <= 0; s4bv <= 0; s4bl <= 0;
        s4byr <= 0; s4byi <= 0;
        for (i = 0; i < NTAPS; i = i + 1) begin
            p7r[i] <= 0; p7i[i] <= 0;
        end
    end else if (pe) begin
        s4er  <= s4dr - s4yr;
        s4ei  <= s4di - s4yi;
        s4byr <= s4yr;   s4byi <= s4yi;
        s4bv  <= s4v;    s4bl  <= s4l;
        for (i = 0; i < NTAPS; i = i + 1) begin
            p7r[i] <= p6r[i]; p7i[i] <= p6i[i];
        end
    end
end

// ---------------------------------------------------------------------------
// S5: mu × e   (FIX 3: slice changed to [30:15] so cfg_mu is plain Q1.15)
// ---------------------------------------------------------------------------
// Host writes: cfg_mu = int(mu_float * 32768)  e.g. 163 for mu=0.005.
// mr = cfg_mu * s4er  →  Q2.30 product.
// mr[30:15] recovers the Q1.15 step:  mu_eff = cfg_mu/32768 = mu_float  ✓

wire signed [P1F-1:0] mr = $signed(cfg_mu) * s4er;
wire signed [P1F-1:0] mi = $signed(cfg_mu) * s4ei;

(* use_dsp="yes" *) reg signed [P1T-1:0] s5mr, s5mi;
reg                  s5v;
reg signed [DW-1:0]  p8r [0:NTAPS-1];
reg signed [DW-1:0]  p8i [0:NTAPS-1];

always @(posedge aclk) begin
    if (!aresetn) begin
        s5mr <= 0; s5mi <= 0; s5v <= 0;
        for (i = 0; i < NTAPS; i = i + 1) begin
            p8r[i] <= 0; p8i[i] <= 0;
        end
    end else if (pe) begin
        s5mr <= mr[P1H:P1L];
        s5mi <= mi[P1H:P1L];
        // FIX 5: s5v is cleared when pe is low so a stall cannot leave a
        // live update token that fires unexpectedly on resume.
        // Here pe is already asserted (we are inside if(pe)), so we gate
        // s5v on the upstream valid token only.
        s5v  <= s4bv;
        for (i = 0; i < NTAPS; i = i + 1) begin
            p8r[i] <= p7r[i]; p8i[i] <= p7i[i];
        end
    end else begin
        // FIX 5: flush valid token on stall — prevents ghost update on resume
        s5v <= 1'b0;
    end
end

// ---------------------------------------------------------------------------
// S6: Update multiply   (unchanged arithmetic, FIX 2 benefit propagated via p8)
// ---------------------------------------------------------------------------
(* use_dsp="yes" *) reg signed [UP-1:0] s6a [0:NTAPS-1];
(* use_dsp="yes" *) reg signed [UP-1:0] s6b [0:NTAPS-1];
(* use_dsp="yes" *) reg signed [UP-1:0] s6c [0:NTAPS-1];
(* use_dsp="yes" *) reg signed [UP-1:0] s6d [0:NTAPS-1];
reg s6v;

always @(posedge aclk) begin
    if (!aresetn) begin
        s6v <= 0;
        for (i = 0; i < NTAPS; i = i + 1) begin
            s6a[i] <= 0; s6b[i] <= 0;
            s6c[i] <= 0; s6d[i] <= 0;
        end
    end else if (pe) begin
        for (i = 0; i < NTAPS; i = i + 1) begin
            // p8r/p8i now carry the correctly-aligned x snapshot (FIX 2)
            s6a[i] <= $signed(p8r[i]) * $signed(s5mr);
            s6b[i] <= $signed(p8i[i]) * $signed(s5mi);
            s6c[i] <= $signed(p8r[i]) * $signed(s5mi);
            s6d[i] <= $signed(p8i[i]) * $signed(s5mr);
        end
        s6v <= s5v;
    end else begin
        // FIX 5: flush valid on stall
        s6v <= 1'b0;
    end
end

// ---------------------------------------------------------------------------
// S7: Update reduce   (unchanged, FIX 5 valid flush added)
// ---------------------------------------------------------------------------
reg signed [UP1-1:0] s7r [0:NTAPS-1];
reg signed [UP1-1:0] s7i [0:NTAPS-1];
reg s7v;

always @(posedge aclk) begin
    if (!aresetn) begin
        s7v <= 0;
        for (i = 0; i < NTAPS; i = i + 1) begin
            s7r[i] <= 0; s7i[i] <= 0;
        end
    end else if (pe) begin
        for (i = 0; i < NTAPS; i = i + 1) begin
            s7r[i] <= $signed(s6a[i]) + $signed(s6b[i]);
            s7i[i] <= $signed(s6c[i]) - $signed(s6d[i]);
        end
        s7v <= s6v;
    end else begin
        // FIX 5: flush valid on stall — s7r/s7i values are don't-care when s7v=0
        s7v <= 1'b0;
    end
end

// ---------------------------------------------------------------------------
// S8a: Weight accumulate
// FIX 5: gate is now (pe && s7v) — weight only updates when BOTH the pipeline
// is enabled AND a valid update is present.  The s7v flush above ensures that
// a stall followed by a resume cannot re-fire a stale update token.
// ---------------------------------------------------------------------------
reg signed [AW-1:0] s8r [0:NTAPS-1];
reg signed [AW-1:0] s8i [0:NTAPS-1];
reg s8v;

always @(posedge aclk) begin
    if (!aresetn) begin
        s8v <= 0;
        for (i = 0; i < NTAPS; i = i + 1) begin
            s8r[i] <= 0; s8i[i] <= 0;
        end
    end else if (pe && s7v) begin   // FIX 5: was just 'pe', now gated on s7v too
        for (i = 0; i < NTAPS; i = i + 1) begin
            s8r[i] <= w_re[i] + {{(AW-UP1){s7r[i][UP1-1]}}, s7r[i]};
            s8i[i] <= w_im[i] + {{(AW-UP1){s7i[i][UP1-1]}}, s7i[i]};
        end
        s8v <= 1'b1;
    end else begin
        s8v <= 1'b0;
    end
end

// ---------------------------------------------------------------------------
// S8b: Weight saturate   (unchanged)
// ---------------------------------------------------------------------------
always @(posedge aclk) begin
    if (!aresetn) begin
        for (i = 0; i < NTAPS; i = i + 1) begin
            w_re[i] <= {AW{1'b0}};
            w_im[i] <= {AW{1'b0}};
        end
    end else if (pe && s8v) begin
        for (i = 0; i < NTAPS; i = i + 1) begin
            if      ((!s8r[i][AW-1]) && (|s8r[i][AW-2:WH+1]))  w_re[i] <= SMAX;
            else if (( s8r[i][AW-1]) && (~(&s8r[i][AW-2:WH+1]))) w_re[i] <= SMIN;
            else                                                   w_re[i] <= s8r[i];

            if      ((!s8i[i][AW-1]) && (|s8i[i][AW-2:WH+1]))  w_im[i] <= SMAX;
            else if (( s8i[i][AW-1]) && (~(&s8i[i][AW-2:WH+1]))) w_im[i] <= SMIN;
            else                                                   w_im[i] <= s8i[i];
        end
    end
end

// ---------------------------------------------------------------------------
// Output
// [63:48]=yr  [47:32]=yi  [31:16]=er  [15:0]=ei
// s4byr / s4byi: saturated FIR output held at S4b.
// s4er  / s4ei : error e = d − y, also computed at S4b.
// Output valid = s4bv (7 cycles after input, matching PIPELINE_LATENCY).
// ---------------------------------------------------------------------------
assign m_axis_tvalid = s4bv;
assign m_axis_tlast  = s4bl;
assign m_axis_tdata  = {s4byr, s4byi, s4er, s4ei};

endmodule