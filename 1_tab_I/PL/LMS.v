`timescale 1ns / 1ps

module lms_weight_update #(
    parameter DATA_WIDTH  = 16,
    parameter ACCUM_WIDTH = 48
)(
    input  wire                     aclk,
    input  wire                     aresetn,

    input  wire [31:0]              s_axis_tdata,
    input  wire                     s_axis_tvalid,
    output wire                     s_axis_tready,
    input  wire                     s_axis_tlast,

    output wire [31:0]              m_axis_tdata,
    output wire                     m_axis_tvalid,
    input  wire                     m_axis_tready,
    output wire                     m_axis_tlast,

    input  wire [DATA_WIDTH-1:0]    cfg_mu
);

    localparam P1_FULL_W    = 2 * DATA_WIDTH;
    localparam P1_TRUNC_W   = 18;
    localparam P1_TRUNC_HI  = 31;
    localparam P1_TRUNC_LO  = 14;
    localparam P2_FRAC_DROP = 16;
    localparam W_OUT_HI     = P2_FRAC_DROP + DATA_WIDTH - 2;

    localparam [ACCUM_WIDTH-1:0] SAT_MAX =
        { {(ACCUM_WIDTH - W_OUT_HI - 1){1'b0}}, {(W_OUT_HI + 1){1'b1}} };
    localparam [ACCUM_WIDTH-1:0] SAT_MIN =
        { {(ACCUM_WIDTH - W_OUT_HI - 1){1'b1}}, {(W_OUT_HI + 1){1'b0}} };

    assign s_axis_tready = m_axis_tready;
    wire   pipe_en       = m_axis_tready;

    wire signed [DATA_WIDTH-1:0] x_in_w = $signed(s_axis_tdata[31:16]);
    wire signed [DATA_WIDTH-1:0] e_in_w = $signed(s_axis_tdata[15:0]);

    reg signed [DATA_WIDTH-1:0] s1_x;
    reg signed [DATA_WIDTH-1:0] s1_e;
    reg signed [DATA_WIDTH-1:0] s1_mu;
    reg                         s1_valid;
    reg                         s1_last;

    always @(posedge aclk) begin
        if (!aresetn) begin
            s1_x     <= {DATA_WIDTH{1'b0}};
            s1_e     <= {DATA_WIDTH{1'b0}};
            s1_mu    <= {DATA_WIDTH{1'b0}};
            s1_valid <= 1'b0;
            s1_last  <= 1'b0;
        end else if (pipe_en) begin
            s1_x     <= x_in_w;
            s1_e     <= e_in_w;
            s1_mu    <= $signed(cfg_mu);
            s1_valid <= s_axis_tvalid;
            s1_last  <= s_axis_tlast;
        end
    end

    (* use_dsp = "yes" *) reg signed [P1_TRUNC_W-1:0] s2_P1;
                          reg signed [DATA_WIDTH-1:0] s2_x_d;
                          reg                         s2_valid;
                          reg                         s2_last;

    wire signed [P1_FULL_W-1:0] next_P1_full = $signed(s1_mu) * $signed(s1_e);

    always @(posedge aclk) begin
        if (!aresetn) begin
            s2_P1    <= {P1_TRUNC_W{1'b0}};
            s2_x_d   <= {DATA_WIDTH{1'b0}};
            s2_valid <= 1'b0;
            s2_last  <= 1'b0;
        end else if (pipe_en) begin
            s2_P1    <= next_P1_full[P1_TRUNC_HI : P1_TRUNC_LO];
            s2_x_d   <= s1_x;
            s2_valid <= s1_valid;
            s2_last  <= s1_last;
        end
    end

    (* use_dsp = "yes" *) reg signed [ACCUM_WIDTH-1:0] s3_P2;
                          reg                          s3_valid;
                          reg                          s3_last;

    wire signed [33:0] P2_raw = $signed(s2_P1) * $signed(s2_x_d);

    always @(posedge aclk) begin
        if (!aresetn) begin
            s3_P2    <= {ACCUM_WIDTH{1'b0}};
            s3_valid <= 1'b0;
            s3_last  <= 1'b0;
        end else if (pipe_en) begin
            s3_P2    <= {{(ACCUM_WIDTH-34){P2_raw[33]}}, P2_raw};
            s3_valid <= s2_valid;
            s3_last  <= s2_last;
        end
    end

    reg signed [ACCUM_WIDTH-1:0] w_reg;
    reg                          s4_valid;
    reg                          s4_last;

    wire signed [ACCUM_WIDTH-1:0] P2_gated = s3_valid ? s3_P2
                                                      : {ACCUM_WIDTH{1'b0}};
    wire signed [ACCUM_WIDTH-1:0] w_sum_c  = w_reg + P2_gated;

    wire ovf_pos_c = (!w_sum_c[ACCUM_WIDTH-1]) &&
                     (|w_sum_c[ACCUM_WIDTH-2 : W_OUT_HI+1]);
    wire ovf_neg_c = ( w_sum_c[ACCUM_WIDTH-1]) &&
                     (!(&w_sum_c[ACCUM_WIDTH-2 : W_OUT_HI+1]));

    always @(posedge aclk) begin
        if (!aresetn) begin
            w_reg    <= {ACCUM_WIDTH{1'b0}};
            s4_valid <= 1'b0;
            s4_last  <= 1'b0;
        end else if (pipe_en) begin
            s4_valid <= s3_valid;
            s4_last  <= s3_last;
            if      (ovf_pos_c) w_reg <= $signed(SAT_MAX);
            else if (ovf_neg_c) w_reg <= $signed(SAT_MIN);
            else                w_reg <= w_sum_c;
        end
    end

    wire signed [DATA_WIDTH-1:0] w_out =
        {w_reg[ACCUM_WIDTH-1], w_reg[W_OUT_HI:P2_FRAC_DROP]};

    assign m_axis_tvalid = s4_valid;
    assign m_axis_tlast  = s4_last;
    assign m_axis_tdata  = {{(32-DATA_WIDTH){1'b0}}, w_out};

endmodule
