% hpre6GPDSCHIndices Physical downlink shared channel resource element indices
%   [IND,INFO] = hpre6GPDSCHIndices(CARRIER,PDSCH) returns the matrix IND
%   containing 1-based physical downlink shared channel resource element
%   (RE) indices within the carrier resource grid, in linear form. The
%   matrix IND is obtained from TS 38.211 Sections 7.3.1.5 and 7.3.1.6, for
%   the given extended carrier configuration CARRIER and extended downlink
%   shared channel configuration PDSCH. The number of columns in IND is
%   equal to the number of antenna ports configured. This syntax also
%   provides the structural information INFO about the bit capacity, symbol
%   capacity, the DM-RS OFDM symbol locations, and PT-RS OFDM symbol
%   locations associated with the physical downlink shared channel.
%
%   CARRIER is an extended carrier configuration object as described in
%   <a href="matlab:help('pre6GCarrierConfig')"
%   >pre6GCarrierConfig</a> with the following properties:
%
%   SubcarrierSpacing - Subcarrier spacing in kHz (default 15)
%   CyclicPrefix      - Cyclic prefix ('normal' (default), 'extended')
%   NSizeGrid         - Number of resource blocks in carrier resource grid
%                       (default 52)
%   NStartGrid        - Start of carrier resource grid relative to CRB 0
%                       (default 0)
%   NSlot             - Slot number (default 0)
%
%   PDSCH is the extended physical downlink shared channel configuration
%   object as described in <a href="matlab:help('pre6GPDSCHConfig')"
%   >pre6GPDSCHConfig</a> with the following properties:
%
%   NSizeBWP              - Size of the bandwidth part (BWP) in
%                           physical resource blocks (PRBs)
%                           (default [])
%   NStartBWP             - Starting PRB index of BWP relative to
%                           common resource block 0 (CRB 0) (default [])
%                           carrier resource grid
%   ReservedPRB           - Cell array of object(s) containing the reserved
%                           physical resource blocks and OFDM symbols
%                           pattern, as described in <a href="matlab:help('nrPDSCHReservedConfig')">nrPDSCHReservedConfig</a>
%                           with properties:
%       PRBSet    - Reserved PRB indices in BWP (0-based) (default [])
%       SymbolSet - OFDM symbols associated with reserved PRBs over one or
%                   more slots (default [])
%       Period    - Total number of slots in the pattern period (default [])
%   ReservedRE            - Reserved resource element (RE) indices
%                           within BWP (0-based) (default [])
%   Modulation            - Modulation scheme(s) of codeword(s)
%                           ('QPSK' (default), '16QAM', '64QAM', '256QAM', '1024QAM')
%   NumLayers             - Number of transmission layers (1...8)
%                           (default 1)
%   MappingType           - Mapping type of physical downlink shared
%                           channel ('A' (default), 'B')
%   SymbolAllocation      - Symbol allocation of physical downlink shared
%                           channel (default [0 14]). This property is a
%                           two-element vector. First element represents
%                           the start of OFDM symbol in a slot. Second
%                           element represents the number of contiguous
%                           OFDM symbols
%   PRBSet                - Resource block allocation (VRB or PRB indices)
%                           (default 0:51)
%   PRBSetType            - Type of indices used in the PRBSet property
%                           ('VRB' (default), 'PRB')
%   VRBToPRBInterleaving  - Flag to enable VRB to PRB interleaving
%                           (0 (default),1)
%   VRBBundleSize         - Bundle size in resource blocks (RBs)
%                           (default 2)
%   RNTI                  - Radio network temporary identifier (0...65535)
%                           (default 1)
%   DMRS                  - PDSCH-specific DM-RS configuration object, as
%                           described in <a href="matlab:help('nrPDSCHDMRSConfig')">nrPDSCHDMRSConfig</a> with properties:
%       DMRSConfigurationType   - DM-RS configuration type (1 (default), 2)
%       DMRSReferencePoint      - The reference point for the DM-RS
%                                 sequence to subcarrier resource mapping
%                                 ('CRB0' (default), 'PRB0')
%       DMRSTypeAPosition       - Position of first DM-RS OFDM symbol in a
%                                 slot (2 (default), 3)
%       DMRSLength              - Number of consecutive DM-RS OFDM symbols
%                                 (1 (default), 2)
%       DMRSAdditionalPosition  - Maximum number of DM-RS additional
%                                 positions (0...3) (default 0)
%       CustomSymbolSet         - Custom DM-RS symbol locations (0-based)
%                                 (default []). This property is used to
%                                 override the standard defined DM-RS
%                                 symbol locations. Each entry corresponds
%                                 to a single-symbol DM-RS
%       DMRSPortSet             - DM-RS antenna port set (0...11)
%                                 (default []). The default value implies
%                                 that the values are in the range from 0
%                                 to NumLayers-1
%       NumCDMGroupsWithoutData - Number of CDM groups without data (1...3)
%                                 (default 2)
%   EnablePTRS            - Enable or disable the PT-RS configuration
%                           (0 (default), 1). The value of 0 implies PT-RS
%                           is disabled and value of 1 implies PT-RS is
%                           enabled
%   PTRS                  - PDSCH-specific PT-RS configuration object, as
%                           described in <a href="matlab:help('nrPDSCHPTRSConfig')">nrPDSCHPTRSConfig</a> with properties:
%       TimeDensity      - PT-RS time density (1 (default), 2, 4)
%       FrequencyDensity - PT-RS frequency density (2 (default), 4)
%       REOffset         - PT-RS resource element offset
%                          ('00' (default), '01', '10', '11')
%       PTRSPortSet      - PT-RS antenna port set (default []). The default
%                          value implies the value is equal to the lowest
%                          DM-RS antenna port configured
%
%   The output structure INFO contains the following fields:
%   G             - Bit capacity of the PDSCH. This must be the
%                   length of codeword from the DL-SCH transport channel
%   Gd            - Number of resource elements per layer/port
%   DMRSSymbolSet - The OFDM symbol locations in a slot containing DM-RS
%                   (0-based)
%   NREPerPRB     - Number of RE per PRB allocated to PDSCH (not
%                   accounting for any reserved resources)
%   PTRSSymbolSet - The OFDM symbol locations in a slot containing PT-RS
%                   (0-based)
%   PRBSet        - PRBs allocated for PDSCH within the BWP
%
%   IND = hpre6GPDSCHIndices(CARRIER,PDSCH,NAME,VALUE,...) specifies
%   additional options as NAME,VALUE pairs to allow control over the format
%   of the indices:
%
%    'IndexStyle'       - 'index' for linear indices (default)
%                         'subscript' for [subcarrier, symbol, antenna]
%                         subscript row form
%
%    'IndexBase'        - '1based' for 1-based indices (default)
%                         '0based' for 0-based indices
%
%    'IndexOrientation' - 'carrier' for carrier oriented indices (default)
%                         'bwp' for bandwidth part oriented indices
%
%   Example:
%   % Generate the data symbol indices (0-based) in linear index form for
%   % a carrier with 330 resource blocks and 120 kHz subcarrier spacing,
%   % consistent with a 500 MHz bandwidth.
%
%   carrier = pre6GCarrierConfig;
%   carrier.NSizeGrid = 330;
%   carrier.SubcarrierSpacing = 120;
%   pdsch = pre6GPDSCHConfig;
%   ind = hpre6GPDSCHIndices(carrier,pdsch,IndexBase='0based');
%
%   See also hpre6GPDSCH, hpre6GPDSCHDecode, hpre6GPDSCHDMRSIndices,
%   hpre6GPDSCHPTRSIndices, pre6GPDSCHConfig, pre6GCarrierConfig.
function [ind, info] = hpre6GPDSCHIndices(carrier, pdsch, varargin)
    narginchk(2, 8);

    % Validate carrier input
    mustBeA(carrier, 'pre6GCarrierConfig');

    % Validate PDSCH input
    mustBeA(pdsch, 'pre6GPDSCHConfig');

    % Parse options
    fcnName = 'hpre6GPDSCHIndices';
    optNames = {'IndexStyle', 'IndexBase', 'IndexOrientation'};
    opts = nr5g.internal.parseOptions(fcnName, optNames, varargin{:});

    % Get PDSCH indices and dimensionality information
    [ind, info] = nrPDSCHIndices(carrier, pdsch, opts);
end
