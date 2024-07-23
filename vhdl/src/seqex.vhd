entity seqex is
  port (
    a : in    bit;
    b : in    bit;
    c : out   bit
  );
end entity seqex;

architecture behav of seqex is

begin

  process_input_check : process (a, b) is
  begin

    if (a = '1' or b = '1') then
      c <= '1';
    else
      c <= '0';
    end if;

  end process process_input_check;

end architecture behav;
