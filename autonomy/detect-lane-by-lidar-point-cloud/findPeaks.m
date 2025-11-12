function [pkHistVal, pkIdx] = findPeaks(histVal)
    pkIdxTemp = (1:size(histVal, 2))';
    histValTemp = [NaN; histVal'; NaN];
    tempIdx = (1:length(histValTemp)).';

    % keep only the first of any adjacent pairs of equal values (including NaN)
    yFinite = ~isnan(histValTemp);
    iNeq = [1; 1 + find((histValTemp(1:end - 1) ~= histValTemp(2:end)) & (yFinite(1:end - 1) | yFinite(2:end)))];
    tempIdx = tempIdx(iNeq);

    % Take the sign of the first sample derivative
    s = sign(diff(histValTemp(tempIdx)));

    % Find local maxima
    maxIdx = 1 + find(diff(s) < 0);

    % Index into the original index vector without the NaN bookend
    pkIdx = tempIdx(maxIdx) - 1;

    % Fetch the coordinates of the peak
    pkHistVal = histVal(pkIdx);
    pkIdx = pkIdxTemp(pkIdx)';
end
