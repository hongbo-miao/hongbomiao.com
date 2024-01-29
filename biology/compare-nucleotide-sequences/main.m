% https://www.mathworks.com/help/bioinfo/ug/example-sequence-alignment.html

humanHEXA = getgenbank('NM_000520');
mouseHEXA = getgenbank('AK080777');

humanProtein = nt2aa(humanHEXA.Sequence);
mouseProtein = nt2aa(mouseHEXA.Sequence);

humanORFs = seqshoworfs(humanHEXA.Sequence);
mouseORFs = seqshoworfs(mouseHEXA.Sequence);

% Truncate an amino acid sequence to only those amino acids in the protein is to first truncate the nucleotide sequence with indices from the seqshoworfs function.
% The ORF for the human HEXA gene and the ORF for the mouse HEXA were both on the first reading frame.
humanPORF = nt2aa(humanHEXA.Sequence(humanORFs(1).Start(1):humanORFs(1).Stop(1)));
mousePORF = nt2aa(mouseHEXA.Sequence(mouseORFs(1).Start(1):mouseORFs(1).Stop(1)));

% Globally align the trimmed amino acid sequences.
[GlobalScore2, GlobalAlignment2] = nwalign(humanPORF, mousePORF);
seqalignviewer(GlobalAlignment2);
