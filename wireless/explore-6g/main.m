% https://www.mathworks.com/help/5g/ug/getting-started-with-6g-exploration-library.html

% Configure a carrier with a subcarrier spacing of 3840 kHz and 36 resource blocks.
carrier = pre6GCarrierConfig(SubcarrierSpacing = 3840, NSizeGrid = 36);

% Compute the sample rate and transmission bandwidth.
ofdmInfo = hpre6GOFDMInfo(carrier);
sampleRate = ofdmInfo.SampleRate;
disp(['Sample rate = ' num2str(sampleRate / 1e9) ' GHz']);

txBandwidth = carrier.NSizeGrid * 12 * carrier.SubcarrierSpacing * 1e3;
disp(['Transmission bandwidth = ' num2str(txBandwidth / 1e9) ' GHz']);

% Display the number of slots per subframe.
disp(['Slots per subframe = ' num2str(carrier.SlotsPerSubframe)]);

% Display the cyclic prefix lengths of the first slot.
% Because the 6G Exploration Library for 5G Toolbox uses the numerology rules of 5G NR, the first cyclic prefix length is much larger than the others.
% As the subcarrier spacing increases, the first cyclic prefix every half subframe increases relative to the other cyclic prefixes.
disp(['Cyclic prefix lengths (in samples) = ' num2str(ofdmInfo.CyclicPrefixLengths(1:carrier.SymbolsPerSlot))]);

% Resource Grids Larger Than 275 Resource Blocks
% At World Radiocommunication Conference 2023 (WRC-23), ITU agreed to assign a new band 6.425 - 7.125 GHz for the terrestrial component of the International Mobile Telecommunications (IMT) system. This band contains 700 MHz of spectrum and is a potential operating band for the next generation of IMT, which is IMT-2030, also known as 6G.
% To evaluate the performance of a 6G candidate waveform occupying 700 MHz bandwidth and using 120 kHz subcarrier spacing, you need a number of resource blocks greater than 275. You can use the NSizeGrid property of pre6GCarrierConfig to create resource grids with more than 275 resource blocks.
% For example, configure a carrier with the largest number of resource blocks that can be allocated in a 700 MHz bandwidth by using 120 kHz subcarrier spacing and without exceeding a bandwidth occupancy of 0.9. The bandwidth occupancy is the ratio of occupied bandwidth to channel bandwidth. A larger bandwidth occupancy results in more capacity but requires better filtering to avoid emissions outside the channel bandwidth.
bandwidthOccupancy = 0.9;
carrier = pre6GCarrierConfig(SubcarrierSpacing = 120);
channelBandwidth = 700e6;
carrier.NSizeGrid = floor((channelBandwidth / (carrier.SubcarrierSpacing * 1e3) * bandwidthOccupancy) / 12);

% Compute the sample rate and transmission bandwidth.
ofdmInfo = hpre6GOFDMInfo(carrier);
sampleRate = ofdmInfo.SampleRate;
disp(['Sample rate = ' num2str(sampleRate / 1e6) ' MHz']);

txBandwidth = carrier.NSizeGrid * 12 * carrier.SubcarrierSpacing * 1e3;
disp(['Transmission bandwidth = ' num2str(txBandwidth / 1e6) ' MHz']);

% Create a pre6GPDSCHConfig object and configure it to allocate all carrier resource blocks and use 64-QAM modulation.
pdsch = pre6GPDSCHConfig;
pdsch.PRBSet = 0:(carrier.NSizeGrid - 1);
pdsch.Modulation = '64QAM';

% Create a carrier resource grid containing PDSCH and PDSCH DM-RS symbols.
nTxAnts = 1;
txGrid = hpre6GResourceGrid(carrier, nTxAnts);

[ind, indinfo] = hpre6GPDSCHIndices(carrier, pdsch);
cw = randi([0 1], indinfo.G, 1);
sym = hpre6GPDSCH(carrier, pdsch, cw);
txGrid(ind) = sym;

dmrsind = hpre6GPDSCHDMRSIndices(carrier, pdsch);
dmrssym = hpre6GPDSCHDMRS(carrier, pdsch);
txGrid(dmrsind) = dmrssym;

% OFDM-modulate the resource grid.
txWaveform = hpre6GOFDMModulate(carrier, txGrid);

% Plot the spectrum of the OFDM waveform.
scope = spectrumAnalyzer(SampleRate = sampleRate);
scope.Title = "Waveform for 700 MHz channel in upper 6 GHz band";
scope.ChannelMeasurements.Enabled = true;
scope.ChannelMeasurements.Span = 700e6;
scope(txWaveform);
