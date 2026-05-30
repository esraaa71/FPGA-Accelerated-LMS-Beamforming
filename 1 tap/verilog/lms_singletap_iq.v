`timescale 1ns / 1ps
// lms_singletap_iq.v  –  Single-Tap Complex I/Q LMS Filter
//
// Structure follows LMS.v (lms_weight_update) but extended for complex I/Q
// data, taking the complex weight-update math from lms_2tap4.v.
//
// AXI-Stream input  (64-bit): [63:48]=xr  [47:32]=xi  [31:16]=dr  [15:0]=di
// AXI-Stream output (64-bit): [63:48]=yr  [47:32]=yi  [31:16]=er  [15:0]=ei
//
// Complex LMS update (conjugate):
//   y_r  = wr*xr - wi*xi
//   y_i  = wr*xi + wi*xr
//   e_r  = d_r - y_r
//   e_i  = d_i - y_i
//   wr  += mu * (e_r*xr + e_i*xi)
//   wi  += mu * (e_i*xr - e_r*xi)

module lms_singletap_iq #(
    parameter DW = 16,   // data/weight word width
    parameter AW = 40    // weight accumulator width (keeps guard bits)
)(
    input  wire          aclk,
    input  wire          aresetn,

    input  wire [63:0]   s_axis_tdata,
    input  wire          s_axis_tvalid,
    output wire          s_axis_tready,
    input  wire          s_axis_tlast,

    output wire [63:0]   m_axis_tdata,
    output wire          m_axis_tvalid,
    input  wire          m_axis_tready,
    output wire          m_axis_tlast,

    input  wire [DW-1:0] cfg_mu
);

// ── Derived parameters (match lms_2tap4.v conventions) ───────────────────
localparam P1F = 2*DW;        // full mu×e product width      = 32
localparam P1T = 18;          // truncated mu×e width
localparam P1H = 31;          // truncation high bit
localparam P1L = 14;          // truncation low bit
localparam FD  = 16;          // fractional bits in weight acc
localparam WH  = FD + DW - 2; // weight output high bit        = 30
localparam FP  = 2*DW;        // FIR multiply full width       = 32
localparam FP1 = FP + 1;      // after ±combine               = 33
localparam UP  = P1T + DW;    // update multiply width         = 34
localparam UP1 = UP + 1;      // sign-extended update          = 35

// Saturation bounds for weight accumulator (AW=40, WH=30)
localparam signed [AW-1:0] SMAX = {{(AW-WH-1){1'b0}}, {(WH+1){1'b1}}};
localparam signed [AW-1:0] SMIN = {{(AW-WH-1){1'b1}}, {(WH+1){1'b0}}};

// ── Pipeline enable – registered for low skew (from lms_2tap4.v) ─────────
(* max_fanout = 40 *) reg pe;
always @(posedge aclk) begin
    if (!aresetn) pe <= 1'b0;
    else          pe <= m_axis_tready;
end
assign s_axis_tready = pe;

// ── Weight extraction helper (identical to lms_2tap4.v) ──────────────────
function signed [DW-1:0] wx;
    input signed [AW-1:0] a;
    wx = {a[AW-1], a[WH:FD]};
endfunction

// ── Weight accumulators ───────────────────────────────────────────────────
reg signed [AW-1:0] w_re;
reg signed [AW-1:0] w_im;

// ── Registered weight extraction ─────────────────────────────────────────
reg signed [DW-1:0] wr;
reg signed [DW-1:0] wi;

always @(posedge aclk) begin
    if (!aresetn) begin
        wr <= 0;  wi <= 0;
    end else if (pe) begin
        wr <= wx(w_re);
        wi <= wx(w_im);
    end
end

// ── Input word unpacking ──────────────────────────────────────────────────
wire signed [DW-1:0] ixr = $signed(s_axis_tdata[63:48]);
wire signed [DW-1:0] ixi = $signed(s_axis_tdata[47:32]);
wire signed [DW-1:0] idr = $signed(s_axis_tdata[31:16]);
wire signed [DW-1:0] idi = $signed(s_axis_tdata[15:0]);

// ── S1: Input register ────────────────────────────────────────────────────
reg signed [DW-1:0] s1_xr, s1_xi, s1_dr, s1_di;
reg s1v, s1l;

always @(posedge aclk) begin
    if (!aresetn) begin
        s1_xr <= 0;  s1_xi <= 0;
        s1_dr <= 0;  s1_di <= 0;
        s1v   <= 0;  s1l   <= 0;
    end else if (pe) begin
        s1_xr <= ixr;  s1_xi <= ixi;
        s1_dr <= idr;  s1_di <= idi;
        s1v   <= s_axis_tvalid;
        s1l   <= s_axis_tlast;
    end
end

// ── S2: FIR multiply – 4 DSPs ────────────────────────────────────────────
//   s2_rr = wr * xr       (real   × real)
//   s2_ii = wi * xi       (imag   × imag)
//   s2_ri = wr * xi       (real   × imag)
//   s2_ir = wi * xr       (imag   × real)
(* use_dsp="yes" *) reg signed [FP-1:0] s2_rr;
(* use_dsp="yes" *) reg signed [FP-1:0] s2_ii;
(* use_dsp="yes" *) reg signed [FP-1:0] s2_ri;
(* use_dsp="yes" *) reg signed [FP-1:0] s2_ir;
reg signed [DW-1:0] s2_dr, s2_di;
reg signed [DW-1:0] s2_xr, s2_xi;   // carry x forward for weight update
reg s2v, s2l;

always @(posedge aclk) begin
    if (!aresetn) begin
        s2_rr <= 0;  s2_ii <= 0;  s2_ri <= 0;  s2_ir <= 0;
        s2_dr <= 0;  s2_di <= 0;
        s2_xr <= 0;  s2_xi <= 0;
        s2v   <= 0;  s2l   <= 0;
    end else if (pe) begin
        s2_rr <= wr * s1_xr;
        s2_ii <= wi * s1_xi;
        s2_ri <= wr * s1_xi;
        s2_ir <= wi * s1_xr;
        s2_dr <= s1_dr;  s2_di <= s1_di;
        s2_xr <= s1_xr;  s2_xi <= s1_xi;
        s2v   <= s1v;    s2l   <= s1l;
    end
end

// ── S3: Combine FIR output + saturate/truncate ───────────────────────────
//   yr = rr - ii   (33-bit)
//   yi = ri + ir   (33-bit)
reg signed [FP1-1:0] s3_yr_full, s3_yi_full;
reg signed [DW-1:0]  s3_dr, s3_di;
reg signed [DW-1:0]  s3_xr, s3_xi;
reg s3v, s3l;

always @(posedge aclk) begin
    if (!aresetn) begin
        s3_yr_full <= 0;  s3_yi_full <= 0;
        s3_dr <= 0;  s3_di <= 0;
        s3_xr <= 0;  s3_xi <= 0;
        s3v   <= 0;  s3l   <= 0;
    end else if (pe) begin
        s3_yr_full <= $signed(s2_rr) - $signed(s2_ii);
        s3_yi_full <= $signed(s2_ri) + $signed(s2_ir);
        s3_dr <= s2_dr;  s3_di <= s2_di;
        s3_xr <= s2_xr;  s3_xi <= s2_xi;
        s3v   <= s2v;    s3l   <= s2l;
    end
end

// Overflow detection and saturation of FIR output
// FP1=33: sign=[32], guard=[31:30], output=[29:15] (Q15 result)
wire yr_pos_ovf = (!s3_yr_full[FP1-1]) && (|s3_yr_full[FP1-2:30]);
wire yr_neg_ovf = ( s3_yr_full[FP1-1]) && (~(&s3_yr_full[FP1-2:30]));
wire yi_pos_ovf = (!s3_yi_full[FP1-1]) && (|s3_yi_full[FP1-2:30]);
wire yi_neg_ovf = ( s3_yi_full[FP1-1]) && (~(&s3_yi_full[FP1-2:30]));

wire signed [DW-1:0] tyr =
    yr_pos_ovf ? {1'b0, {(DW-1){1'b1}}} :
    yr_neg_ovf ? {1'b1, {(DW-1){1'b0}}} :
                 {s3_yr_full[FP1-1], s3_yr_full[29:15]};

wire signed [DW-1:0] tyi =
    yi_pos_ovf ? {1'b0, {(DW-1){1'b1}}} :
    yi_neg_ovf ? {1'b1, {(DW-1){1'b0}}} :
                 {s3_yi_full[FP1-1], s3_yi_full[29:15]};

// ── S4a: Register saturated FIR output ───────────────────────────────────
reg signed [DW-1:0] s4_yr, s4_yi;
reg signed [DW-1:0] s4_dr, s4_di;
reg signed [DW-1:0] s4_xr, s4_xi;
reg s4v, s4l;

always @(posedge aclk) begin
    if (!aresetn) begin
        s4_yr <= 0;  s4_yi <= 0;
        s4_dr <= 0;  s4_di <= 0;
        s4_xr <= 0;  s4_xi <= 0;
        s4v   <= 0;  s4l   <= 0;
    end else if (pe) begin
        s4_yr <= tyr;  s4_yi <= tyi;
        s4_dr <= s3_dr;  s4_di <= s3_di;
        s4_xr <= s3_xr;  s4_xi <= s3_xi;
        s4v   <= s3v;    s4l   <= s3l;
    end
end

// ── S4b: Error computation  e = d - y ────────────────────────────────────
reg signed [DW-1:0] s4_er,  s4_ei;
reg signed [DW-1:0] s4b_yr, s4b_yi;   // output yr/yi held one more cycle
reg signed [DW-1:0] s4b_xr, s4b_xi;
reg s4bv, s4bl;

always @(posedge aclk) begin
    if (!aresetn) begin
        s4_er  <= 0;  s4_ei  <= 0;
        s4b_yr <= 0;  s4b_yi <= 0;
        s4b_xr <= 0;  s4b_xi <= 0;
        s4bv   <= 0;  s4bl   <= 0;
    end else if (pe) begin
        s4_er  <= s4_dr - s4_yr;
        s4_ei  <= s4_di - s4_yi;
        s4b_yr <= s4_yr;  s4b_yi <= s4_yi;
        s4b_xr <= s4_xr;  s4b_xi <= s4_xi;
        s4bv   <= s4v;    s4bl   <= s4l;
    end
end

// ── S5: mu × e  (2 DSPs) – truncate to 18 bits ───────────────────────────
wire signed [P1F-1:0] mr_full = $signed(cfg_mu) * $signed(s4_er);
wire signed [P1F-1:0] mi_full = $signed(cfg_mu) * $signed(s4_ei);

(* use_dsp="yes" *) reg signed [P1T-1:0] s5_mr, s5_mi;
reg signed [DW-1:0]  s5_xr, s5_xi;
reg s5v;

always @(posedge aclk) begin
    if (!aresetn) begin
        s5_mr <= 0;  s5_mi <= 0;
        s5_xr <= 0;  s5_xi <= 0;
        s5v   <= 0;
    end else if (pe) begin
        s5_mr <= mr_full[P1H:P1L];
        s5_mi <= mi_full[P1H:P1L];
        s5_xr <= s4b_xr;  s5_xi <= s4b_xi;
        s5v   <= s4bv;
    end
end

// ── S6: Update multiply – 4 DSPs ─────────────────────────────────────────
//   delta_wr = mu*(er*xr + ei*xi)  →  s6a=xr*mr,  s6b=xi*mi
//   delta_wi = mu*(ei*xr - er*xi)  →  s6c=xr*mi,  s6d=xi*mr
(* use_dsp="yes" *) reg signed [UP-1:0] s6a;  // xr * mr
(* use_dsp="yes" *) reg signed [UP-1:0] s6b;  // xi * mi
(* use_dsp="yes" *) reg signed [UP-1:0] s6c;  // xr * mi
(* use_dsp="yes" *) reg signed [UP-1:0] s6d;  // xi * mr
reg s6v;

always @(posedge aclk) begin
    if (!aresetn) begin
        s6a <= 0;  s6b <= 0;  s6c <= 0;  s6d <= 0;
        s6v <= 0;
    end else if (pe) begin
        s6a <= $signed(s5_xr) * $signed(s5_mr);
        s6b <= $signed(s5_xi) * $signed(s5_mi);
        s6c <= $signed(s5_xr) * $signed(s5_mi);
        s6d <= $signed(s5_xi) * $signed(s5_mr);
        s6v <= s5v;
    end
end

// ── S7: Update reduce ─────────────────────────────────────────────────────
reg signed [UP1-1:0] s7_r, s7_i;
reg s7v;

always @(posedge aclk) begin
    if (!aresetn) begin
        s7_r <= 0;  s7_i <= 0;
        s7v  <= 0;
    end else if (pe) begin
        s7_r <= $signed(s6a) + $signed(s6b);  // er*xr + ei*xi
        s7_i <= $signed(s6c) - $signed(s6d);  // ei*xr - er*xi
        s7v  <= s6v;
    end
end

// ── S8: Weight accumulate + saturate (single-cycle feedback) ────────────
// Combinational add + overflow check, registered in ONE always block.
// This ensures every arriving update is applied with no dropped cycles.
wire signed [AW-1:0] w_re_next = w_re + (s7v ? {{(AW-UP1){s7_r[UP1-1]}}, s7_r} : {AW{1'b0}});
wire signed [AW-1:0] w_im_next = w_im + (s7v ? {{(AW-UP1){s7_i[UP1-1]}}, s7_i} : {AW{1'b0}});

wire re_pos_ovf = (!w_re_next[AW-1]) && ( |w_re_next[AW-2:WH+1]);
wire re_neg_ovf = ( w_re_next[AW-1]) && (~&w_re_next[AW-2:WH+1]);
wire im_pos_ovf = (!w_im_next[AW-1]) && ( |w_im_next[AW-2:WH+1]);
wire im_neg_ovf = ( w_im_next[AW-1]) && (~&w_im_next[AW-2:WH+1]);

always @(posedge aclk) begin
    if (!aresetn) begin
        w_re <= {AW{1'b0}};
        w_im <= {AW{1'b0}};
    end else if (pe) begin
        // Real weight
        if      (re_pos_ovf) w_re <= SMAX;
        else if (re_neg_ovf) w_re <= SMIN;
        else                 w_re <= w_re_next;
        // Imaginary weight
        if      (im_pos_ovf) w_im <= SMAX;
        else if (im_neg_ovf) w_im <= SMIN;
        else                 w_im <= w_im_next;
    end
end

// ── Output ────────────────────────────────────────────────────────────────
// [63:48] = yr   [47:32] = yi   [31:16] = er   [15:0] = ei
assign m_axis_tvalid = s4bv;
assign m_axis_tlast  = s4bl;
assign m_axis_tdata  = {s4b_yr, s4b_yi, s4_er, s4_ei};

endmodule
