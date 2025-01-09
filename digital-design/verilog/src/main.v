module counter_4bit (
    input  wire       clk,   // Clock input
    input  wire       rst,   // Reset input
    output reg  [3:0] count  // 4-bit counter output
);

  // Sequential logic block
  always @(posedge clk) begin
    if (rst) begin
      // Synchronous reset
      count <= 4'b0000;
    end else begin
      // Increment counter
      count <= count + 1;
    end
  end
endmodule

// Test bench
module main;
  // Test bench signals
  reg clk;
  reg rst;
  wire [3:0] count;

  // Instantiate the counter
  counter_4bit counter_inst (
      .clk  (clk),
      .rst  (rst),
      .count(count)
  );

  // Clock generation
  initial begin
    clk = 0;
    forever #5 clk = ~clk;
  end

  // Test stimulus
  initial begin
    // Initialize
    rst = 1;

    // Wait for 2 clock cycles
    #20;

    // Release reset
    rst = 0;

    // Let it count for a while
    #160;

    // Apply reset again
    rst = 1;
    #20;

    // End simulation
    $finish;
  end

  // Monitor changes
  initial begin
    $monitor("Time=%0t rst=%b count=%b", $time, rst, count);
  end
endmodule
