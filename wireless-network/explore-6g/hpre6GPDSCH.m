% hpre6GPDSCH Physical downlink shared channel
%   SYM = hpre6GPDSCH(CARRIER,PDSCH,CWS) returns a complex matrix SYM
%   containing the physical downlink shared channel modulation symbols as
%   defined in TS 38.211 Sections 7.3.1.1 - 7.3.1.3, given the extended
%   carrier configuration CARRIER, extended downlink shared channel
%   configuration PDSCH, and DL-SCH codeword(s) CWS.
%
%   CARRIER is an extended carrier configuration object as described in
%   <a href="matlab:help('pre6GCarrierConfig')"
%   >pre6GCarrierConfig</a> with the following properties:
%      NCellID - Physical layer cell identity (0...1007) (default 1)
%
%   PDSCH is the extended physical downlink shared channel configuration
%   object as described in <a href="matlab:help('pre6GPDSCHConfig')"
%   >pre6GPDSCHConfig</a> with the following properties:
%      Modulation - Modulation scheme(s) of codeword(s)
%                   ('QPSK' (default), '16QAM', '64QAM', '256QAM', '1024QAM')
%      NumLayers  - Number of transmission layers (1...8) (default 1)
%      NID        - PDSCH scrambling identity (0...1023) (default []). Use
%                   empty ([]) to set the value to NCellID
%      RNTI       - Radio network temporary identifier (0...65535)
%                   (default 1)
%
%   CWS represents one or two DL-SCH codewords as described in TS 38.212
%   Section 7.2.6. CWS can be a column vector (representing one codeword)
%   or a cell array of one or two column vectors (representing one or two
%   codewords).
%
%   Example:
%   % Generate PDSCH symbols for a single codeword of 8000 bits, using
%   % 256QAM modulation and 4 layers
%
%   carrier = pre6GCarrierConfig;
%   carrier.NCellID = 42;
%   pdsch = pre6GPDSCHConfig;
%   pdsch.Modulation = '256QAM';
%   pdsch.NumLayers = 4;
%   pdsch.RNTI = 6143;
%   data = randi([0 1],8000,1);
%   txsym = hpre6GPDSCH(carrier,pdsch,data);
%
%   See also hpre6GPDSCHDecode, pre6GCarrierConfig, pre6GPDSCHConfig.
function sym = hpre6GPDSCH(carrier, pdsch, cws)
    narginchk(3, 3);

    % Validate carrier input
    mustBeA(carrier, 'pre6GCarrierConfig');

    % Validate PDSCH input
    mustBeA(pdsch, 'pre6GPDSCHConfig');

    % Perform PDSCH modulation
    sym = nrPDSCH(carrier, pdsch, cws, OutputDataType = 'single');
end
