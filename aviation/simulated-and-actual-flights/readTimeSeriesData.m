function [xyz, ptp] = readTimeSeriesData(varargin)
    h = varargin{end};
    if size(varargin, 2) == 2
        t = varargin{1};
        if isfinite(varargin{1})
            xyz(1) = interp1(h.TimeSeriesSource.time, h.TimeSeriesSource.X, t);
            xyz(2) = interp1(h.TimeSeriesSource.time, h.TimeSeriesSource.Y, t);
            xyz(3) = interp1(h.TimeSeriesSource.time, h.TimeSeriesSource.Z, t);

            ptp(1) = interp1(h.TimeSeriesSource.time, h.TimeSeriesSource.phi, t);
            ptp(2) = interp1(h.TimeSeriesSource.time, h.TimeSeriesSource.theta, t);
            ptp(3) = interp1(h.TimeSeriesSource.time, h.TimeSeriesSource.psi, t);
        else
            xyz = h.TimeSeriesSource.time(1);
            ptp = h.TimeSeriesSource.time(end);
        end
    else
        xyz = [h.TimeSeriesSource.X(1), ...
               h.TimeSeriesSource.Y(1), ...
               h.TimeSeriesSource.Z(1)];
        ptp = [h.TimeSeriesSource.phi(1), ...
               h.TimeSeriesSource.theta(1), ...
               h.TimeSeriesSource.psi(1)];
    end
