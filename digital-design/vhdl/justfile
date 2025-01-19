ghdl-install:
	brew install --cask ghdl
gtkwave-install:
	brew install --cask gtkwave

ghdl-import:
	ghdl import src/seqex.vhd
	ghdl import src/t_seqex.vhd
ghdl-make:
	ghdl make t_seqex
ghdl-run:
	ghdl run t_seqex --wave=t_seqex.ghw
