% hpre6GOFDMInfo OFDM modulation related information
%   INFO = hpre6GOFDMInfo(CARRIER) provides dimensional information related
%   to OFDM modulation, given extended carrier configuration object
%   CARRIER.
%
%   CARRIER is an extended carrier configuration object, <a
%   href="matlab:help('pre6GCarrierConfig')"
%   >pre6GCarrierConfig</a>.
%   Only these object properties are relevant for this function:
%
%   SubcarrierSpacing - Subcarrier spacing in kHz (default 15)
%   CyclicPrefix      - Cyclic prefix ('normal', 'extended')
%   NSizeGrid         - Number of resource blocks in carrier resource grid
%                       (default 52)
%
%   INFO is a structure containing the fields:
%
%   Nfft                - Number of IFFT points used in the OFDM modulator
%   SampleRate          - Sample rate of the OFDM modulated waveform
%   CyclicPrefixLengths - Cyclic prefix length (in samples) of each OFDM
%                         symbol in a subframe, starting at slot 0
%   SymbolLengths       - Total length (in samples) of each OFDM symbol in
%                         a subframe, including the cyclic prefix and
%                         starting at slot 0
%   Windowing           - Number of time-domain samples over which
%                         windowing and overlapping of OFDM symbols is
%                         applied
%   SymbolPhases        - Phase precompensation applied for each OFDM
%                         symbol due to the phase term per OFDM symbol in
%                         TS 38.211 Section 5.4. <a
%                         href="matlab:help('nrOFDMModulate')"
%                         >nrOFDMModulate</a> applies
%                         this precompensation during modulation and
%                         <a href="matlab:help('nrOFDMDemodulate')"
%                         >nrOFDMDemodulate</a> performs decompensation
%                         during demodulation.
%   SymbolsPerSlot      - Number of OFDM symbols in a slot
%   SlotsPerSubframe    - Number of slots in a 1 ms subframe
%   SlotsPerFrame       - Number of slots in a 10 ms frame
%
%   Note that the number of samples in the INFO.CyclicPrefixLengths,
%   INFO.SymbolLengths, and INFO.Windowing fields apply to the sample rate
%   of the IFFT of size INFO.Nfft used during OFDM symbol construction.
%   This may be different from the sample rate of the waveform in the case
%   that the 'SampleRate' NAME,VALUE pair below is specified. Note also
%   that the IFFT size can be specified using the 'Nfft' NAME,VALUE pair.
%
%   INFO = hpre6GOFDMInfo(...,NAME,VALUE) specifies additional options as
%   NAME,VALUE pairs to allow control over the OFDM modulation:
%
%   Nfft                - Desired number of IFFT points to use in the OFDM
%                         modulator. If absent or set to [], a default
%                         value is selected based on other parameters, see
%                         <a href="matlab: doc('nrOFDMModulate')"
%                         >nrOFDMModulate</a> for details
%   Windowing           - Number of time-domain samples over which
%                         windowing and overlapping of OFDM symbols is
%                         applied. If absent or set to [], a default value
%                         is selected based on other parameters, see
%                         <a href="matlab: doc('nrOFDMModulate')"
%                         >nrOFDMModulate</a> for details
%   CarrierFrequency    - Carrier frequency (in Hz) to calculate the phase
%                         precompensation applied for each OFDM symbol
%                         (denoted f_0 in TS 38.211 Section 5.4). Default
%                         is 0
%
%   Note that for the numerologies specified in TS 38.211 Section 4.2,
%   extended cyclic prefix length is only applicable for 60 kHz subcarrier
%   spacing. Note that the number of samples specified in the 'Windowing'
%   option applies to the IFFT of size INFO.Nfft used during OFDM symbol
%   construction.
%
%   Example:
%   % Create OFDM information related to a carrier with 330 resource blocks
%   % and 120 kHz subcarrier spacing, consistent with a 500 MHz bandwidth
%
%   % Configure carrier
%   carrier = pre6GCarrierConfig;
%   carrier.NSizeGrid = 330;
%   carrier.SubcarrierSpacing = 120;
%
%   % Create OFDM information
%   info = hpre6GOFDMInfo(carrier)
%
%   See also pre6GCarrierConfig, hpre6GOFDMInfo, hpre6GOFDMModulate,
%   hpre6GResourceGrid.
function info = hpre6GOFDMInfo(carrier, varargin)
    narginchk(1, 7);

    % Validate carrier input
    mustBeA(carrier, 'pre6GCarrierConfig');

    % Parse options
    fcnName = 'hpre6GOFDMInfo';
    optNames = {'Nfft', 'Windowing', 'CarrierFrequency'};
    opts = nr5g.internal.parseOptions(fcnName, optNames, varargin{:});

    % Get OFDM information
    info = nrOFDMInfo(carrier, opts);
end
