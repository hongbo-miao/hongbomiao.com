entity t_seqex is
end entity t_seqex;

architecture test of t_seqex is

  component seqex is
    port (
      a : in    bit;
      b : in    bit;
      c : out   bit
    );
  end component seqex;

  signal a : bit;
  signal b : bit;
  signal c : bit;

begin

  uut : component seqex
    port map (
      a => a,
      b => b,
      c => c
    );

  a <= '1' after 5 ns, '0' after 10 ns;

end architecture test;
