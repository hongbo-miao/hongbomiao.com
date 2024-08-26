% hpre6GResourceGrid Carrier slot resource grid
%   GRID = hpre6GResourceGrid(CARRIER,P) returns an empty carrier slot
%   resource grid, as a complex array of all zeros, for the number of
%   antennas, P, and the specified extended carrier configuration, CARRIER.
%
%   CARRIER is an extended carrier configuration object, <a
%   href="matlab:help('pre6GCarrierConfig')"
%   >pre6GCarrierConfig</a>.
%   Only these object properties are relevant for this function:
%
%   CyclicPrefix - Cyclic prefix ('normal', 'extended')
%   NSizeGrid    - Number of resource blocks in carrier resource grid
%                  (default 52)
%
%   GRID is a complex K-by-L-by-P array of zeros, where K is the number of
%   subcarriers, L is the number of OFDM symbols and P is the number of
%   antennas.
%
%   Example:
%   % Create a carrier resource grid for 330 resource blocks and 2 antennas
%
%   % Configure carrier for 330 resource blocks
%   carrier = pre6GCarrierConfig;
%   carrier.NSizeGrid = 330;
%   carrier.SubcarrierSpacing = 120;
%
%   % Create carrier resource grid for 2 antennas
%   grid = hpre6GResourceGrid(carrier,2);
%   size(grid)
%
%   See also pre6GCarrierConfig, hpre6GOFDMModulate, hpre6GOFDMInfo,
%   hpre6GOFDMDemodulate.
function grid = hpre6GResourceGrid(carrier, P)
    narginchk(2, 2);

    % Validate carrier input
    mustBeA(carrier, 'pre6GCarrierConfig');

    % Create resource grid
    K = carrier.NSizeGrid * 12;
    L = carrier.SymbolsPerSlot;
    grid = zeros([K L P], 'like', single(1i));
end
