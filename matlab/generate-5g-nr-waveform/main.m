% https://www.mathworks.com/help/5g/gs/5g-nr-waveform-generation.html

waveconfig = nrDLCarrierConfig;

% Create a first instance of the PDSCH configuration object and set the allocation to span PRBs 0 to 10.
pdschConfig1 = nrWavegenPDSCHConfig;
pdschConfig1.PRBSet = 0:10;
waveconfig.PDSCH{1} = pdschConfig1;

% Create a second instance of the PDSCH configuration object and set the allocation to span PRBs 40 to 50 and OFDM symbols 2 to 10.
pdschConfig2 = nrWavegenPDSCHConfig;
pdschConfig2.PRBSet = 40:50;
pdschConfig2.SymbolAllocation = [2 10];

waveconfig.PDSCH{2} = pdschConfig2;
[waveform, waveformInfo] = nrWaveformGenerator(waveconfig);

% Plot spectrogram of waveform for first antenna port
samplerate = waveformInfo.ResourceGrids(1).Info.SampleRate;
nfft = waveformInfo.ResourceGrids(1).Info.Nfft;
figure;
spectrogram(waveform(:, 1), ones(nfft, 1), 0, nfft, 'centered', samplerate, 'yaxis', 'MinThreshold', -130);
title('Spectrogram of 5G Downlink Baseband Waveform');
