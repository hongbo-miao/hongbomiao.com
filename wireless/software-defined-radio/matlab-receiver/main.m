stationFrequencyHz = 94.9e6;
frontEndSampleRateHz = 240e3;  % RTL-SDR rate; 240 kHz / 5 lands on 48 kHz audio
audioSampleRateHz = 48e3;
tunerGainDb = 40;
runTimeSeconds = 300;

% One frame is 1/20 s of I/Q. 12000 is a multiple of the 240k/48k = 5
% decimation, which comm.FMBroadcastDemodulator requires.
framesPerSecond = 20;
samplesPerFrame = frontEndSampleRateHz / framesPerSecond;

rtlsdrReceiver = comm.SDRRTLReceiver( ...
                                     CenterFrequency = stationFrequencyHz, ...
                                     EnableTunerAGC = false, ...
                                     TunerGain = tunerGainDb, ...
                                     SampleRate = frontEndSampleRateHz, ...
                                     SamplesPerFrame = samplesPerFrame, ...
                                     OutputDataType = "single");

fmDemodulator = comm.FMBroadcastDemodulator( ...
                                            SampleRate = frontEndSampleRateHz, ...
                                            FrequencyDeviation = 75e3, ...
                                            FilterTimeConstant = 75e-6, ...
                                            AudioSampleRate = audioSampleRateHz, ...
                                            Stereo = false);

audioPlayer = audioDeviceWriter(SampleRate = audioSampleRateHz);

% Real-time RF spectrum of the raw I/Q. CenterFrequency shifts the x-axis so
% the carrier appears at its true frequency (94.9 MHz) instead of baseband
% 0 Hz; the 240 kHz span shows the FM channel plus its neighbors.
rfSpectrumScope = spectrumAnalyzer( ...
                                   SampleRate = frontEndSampleRateHz, ...
                                   CenterFrequency = stationFrequencyHz, ...
                                   Title = "RTL-SDR FM spectrum");

% Release the dongle, demodulator, audio device and scope no matter how the
% script exits (including Ctrl-C), so the radio is freed for the next run.
cleanup = onCleanup(@() releaseObjects( ...
                                       {rtlsdrReceiver, fmDemodulator, audioPlayer, rfSpectrumScope}));

fprintf("Receiving wbfm at %.3f MHz, %d Hz audio (%d s)\n", ...
        stationFrequencyHz / 1e6, audioSampleRateHz, runTimeSeconds);

for frameIndex = 1:(framesPerSecond * runTimeSeconds)
    iqSamples = rtlsdrReceiver();
    rfSpectrumScope(iqSamples);
    audioSamples = fmDemodulator(iqSamples);
    audioPlayer(audioSamples);
end

fprintf("Stopping receiver\n");

function releaseObjects(objects)
    for index = 1:numel(objects)
        release(objects{index});
    end
end
