function [histVal, yvals] = computeHistogram(ptCloud, histogramBinResolution)
    numBins = ceil((ptCloud.YLimits(2) - ptCloud.YLimits(1)) / histogramBinResolution);
    histVal = zeros(1, numBins - 1);
    binStartY = linspace(ptCloud.YLimits(1), ptCloud.YLimits(2), numBins);
    yvals = zeros(1, numBins - 1);
    for i = 1:numBins - 1
        roi = [-inf 15 binStartY(i) binStartY(i + 1) -inf inf];
        ind = findPointsInROI(ptCloud, roi);
        subPc = select(ptCloud, ind);
        if subPc.Count
            histVal(i) = sum(subPc.Intensity);
            yvals(i) = (binStartY(i) + binStartY(i + 1)) / 2;
        end
    end
end
