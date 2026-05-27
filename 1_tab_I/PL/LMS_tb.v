`timescale 1ns / 1ps

module tb_lms_adaptive_demo;

    parameter      CLK_PERIOD  = 10;
    parameter      NUM_SAMPLES = 5000;
    localparam real PI         = 3.14159265358979;
    localparam real SINE_FREQ  = 0.01;
    localparam real SINE_AMP   = 0.5;
    localparam real NOISE_AMP  = 0.3;
    localparam real Q15_SCALE  = 32768.0;
    localparam [15:0] MU_Q15   = 16'h0666;

    reg aclk;
    reg aresetn;

    initial aclk = 1'b0;
    always #(CLK_PERIOD/2) aclk = ~aclk;

    reg  [31:0] s_axis_tdata;
    reg         s_axis_tvalid;
    wire        s_axis_tready;
    reg         s_axis_tlast;

    wire [31:0] m_axis_tdata;
    wire        m_axis_tvalid;
    reg         m_axis_tready;
    wire        m_axis_tlast;

    reg  [15:0] cfg_mu;

    lms_weight_update #(
        .DATA_WIDTH (16),
        .ACCUM_WIDTH(48)
    ) dut (
        .aclk          (aclk),
        .aresetn       (aresetn),
        .s_axis_tdata  (s_axis_tdata),
        .s_axis_tvalid (s_axis_tvalid),
        .s_axis_tready (s_axis_tready),
        .s_axis_tlast  (s_axis_tlast),
        .m_axis_tdata  (m_axis_tdata),
        .m_axis_tvalid (m_axis_tvalid),
        .m_axis_tready (m_axis_tready),
        .m_axis_tlast  (m_axis_tlast),
        .cfg_mu        (cfg_mu)
    );

    real desired_real;
    real noise_real;
    real input_msg_real;
    real y_real;
    real output_msg_real;
    real weight_real;
    real error_mag_real;

    reg signed [15:0] x_q15;
    reg signed [15:0] e_q15;
    reg signed [15:0] w_q15;

    integer sample_count;
    integer file_handle;
    integer rand_seed;

    always @(posedge aclk) begin
        if (!aresetn)
            w_q15 <= 16'sd0;
        else if (m_axis_tvalid && m_axis_tready)
            w_q15 <= $signed(m_axis_tdata[15:0]);
    end

    always @* begin
        weight_real = $itor($signed(w_q15)) / Q15_SCALE;
    end

    initial begin
        aresetn         = 1'b0;
        s_axis_tvalid   = 1'b0;
        s_axis_tlast    = 1'b0;
        s_axis_tdata    = 32'h0;
        m_axis_tready   = 1'b1;
        cfg_mu          = MU_Q15;
        x_q15           = 16'sd0;
        e_q15           = 16'sd0;
        sample_count    = 0;
        rand_seed       = 32'd1;
        desired_real    = 0.0;
        noise_real      = 0.0;
        input_msg_real  = 0.0;
        y_real          = 0.0;
        output_msg_real = 0.0;
        error_mag_real  = 0.0;

        file_handle = $fopen("lms_log.csv", "w");
        $fdisplay(file_handle, "sample,desired,noise,input_msg,output_msg,weight");

        $display("=========================================================");
        $display("  LMS Adaptive Noise Cancellation Demo");
        $display("  Sine amp = %0f, Noise amp = %0f", SINE_AMP, NOISE_AMP);
        $display("  mu       = 0.05  (Q1.15 = 0x%04h)", MU_Q15);
        $display("  Samples  = %0d", NUM_SAMPLES);
        $display("=========================================================");

        repeat (10) @(posedge aclk);
        #1;
        aresetn = 1'b1;
        repeat (4) @(posedge aclk);
        #1;

        while (sample_count < NUM_SAMPLES) begin

            desired_real = SINE_AMP * $sin(2.0 * PI * SINE_FREQ * sample_count);

            noise_real = NOISE_AMP *
                         ($itor($random(rand_seed)) / 2147483648.0);

            input_msg_real = desired_real + noise_real;

            y_real = weight_real * noise_real;

            output_msg_real = input_msg_real - y_real;

            error_mag_real = output_msg_real - desired_real;
            if (error_mag_real < 0.0) error_mag_real = -error_mag_real;

            x_q15 = $rtoi(noise_real      * Q15_SCALE);
            e_q15 = $rtoi(output_msg_real * Q15_SCALE);

            $fdisplay(file_handle, "%0d,%f,%f,%f,%f,%f",
                      sample_count, desired_real, noise_real,
                      input_msg_real, output_msg_real, weight_real);

            if ((sample_count % 500) == 0)
                $display("[sample %0d]  weight = %0f   |err| = %0f",
                         sample_count, weight_real, error_mag_real);

            s_axis_tdata  = {x_q15, e_q15};
            s_axis_tvalid = 1'b1;
            s_axis_tlast  = (sample_count == NUM_SAMPLES - 1);

            @(posedge aclk);
            while (!s_axis_tready) @(posedge aclk);
            #1;

            sample_count = sample_count + 1;
        end

        s_axis_tvalid = 1'b0;
        s_axis_tlast  = 1'b0;

        repeat (10) @(posedge aclk);

        $fclose(file_handle);
        $display("=========================================================");
        $display("  DONE.   final weight = %0f   (target 1.0)", weight_real);
        $display("          final |err|  = %0f", error_mag_real);
        $display("  CSV log: lms_log.csv");
        $display("=========================================================");
        $finish;
    end

    initial begin
        #(CLK_PERIOD * (NUM_SAMPLES + 1000));
        $display("[ERROR] watchdog timeout");
        $finish;
    end

endmodule
