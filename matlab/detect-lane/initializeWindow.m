function [yval, detectedPeaks] = initializeWindow(yvals, peaks, laneWidth)
    leftLanesIndices = yvals >= 0;
    rightLanesIndices = yvals < 0;
    leftLaneYs = yvals(leftLanesIndices);
    rightLaneYs = yvals(rightLanesIndices);
    peaksLeft = peaks(leftLanesIndices);
    peaksRight = peaks(rightLanesIndices);
    diff = zeros(sum(leftLanesIndices), sum(rightLanesIndices));
    for i = 1:sum(leftLanesIndices)
        for j = 1:sum(rightLanesIndices)
            diff(i, j) = abs(laneWidth - (leftLaneYs(i) - rightLaneYs(j)));
        end
    end
    [~, minIndex] = min(diff(:));
    [row, col] = ind2sub(size(diff), minIndex);
    yval = [leftLaneYs(row) rightLaneYs(col)];
    detectedPeaks = [peaksLeft(row) peaksRight(col)];
    estimatedLaneWidth = leftLaneYs(row) - rightLaneYs(col);

    % If the calculated lane width is not within the bounds,
    % return the lane with highest peak
    if abs(estimatedLaneWidth - laneWidth) > 0.5
        if max(peaksLeft) > max(peaksRight)
            yval = [leftLaneYs(maxLeftInd) NaN];
            detectedPeaks = [peaksLeft(maxLeftInd) NaN];
        else
            yval = [NaN rightLaneYs(maxRightInd)];
            detectedPeaks = [NaN rightLaneYs(maxRightInd)];
        end
    end
end
