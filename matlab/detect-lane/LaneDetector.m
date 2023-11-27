% Detects lanes points on lidar point clouds by detecting peaks in the intensities of lidar data.
% The detected lane points are also fitted with a second degree polynomial.
%
% laneDetector = LaneDetector(Name,Value) specifies additional
% name-value pair arguments as described below:
%
%   'ROI'                       The roi is a cuboid specified as a
%                               [xmin, xmax, ymin, ymax, zmin, zmax].
%                               It is used to define the region of
%                               interest for possible lane points.
%
%                               Default: [0 40 -3, 3 -4, 1]
%
%   'LaneWidth'                 The lateral distance between the two
%                               lanes in world units.
%
%                               Default: 4
%
%   'HistogramBinResolution'    The resolution to compute the histogram
%                               of intensities in world units.
%
%                               Default: 0.2
%
%   'VerticalBinResolution'     The vertical bin resolution represents
%                               the size of bins in world units along
%                               ego vehicle direction for updating the
%                               next window in sliding window algorithm.
%                               If the distance between discontinous lane
%                               markings is more, this parameter can be
%                               increased to improve the detection.
%
%                               Default: 1
%
%   'HorizontalBinResolution'   The horizontal bin resolution represents
%                               the size of bins in world units
%                               perpendicular to the ego vehicle
%                               direction for updating the next window
%                               in sliding window algorithm. If the
%                               lane markings are broad, this parameter
%                               can be increased to improve the detection.
%
%                               Default: 1
%
% LaneDetector methods :
%
% detectLanes                 Detects ego vehicle lanes as left lanes
%                             and right lanes stored in a cell array in
%                             the same order.
%
% updateLanePolynomial        Updates the detected lane points using a
%                             polynomial defined in X-Y coordinates.

classdef LaneDetector < handle

    properties (Access = public)
        % Defines plane model
        PlaneModel

        % Defines distance between ego lanes
        LaneWidth

        % Defines detected lane parameters
        LanePolynomial
    end

    properties (Access = private)
        % Region of Interest
        ROI

        % Lane start points
        StartLanePoints

        % Resolution of windows defined along lane axis
        VerticalBinResolution

        % Resolution of windows defined along horizontal axis
        HorizontalBinResolution

        % Bin Resolution for histogram creation
        HistogramBinResolution

        % Number of lanes to detect
        LanesDetect
    end

    properties (Constant, Access = protected)
        DefaultROI = [0 40 -3, 3 -4, 1]
        DefaultLaneWidth = 4
        DefaultVerticalBinResolution = 1
        DefaultHorizontalBinResolution = 1
        DefaultHistogramBinResolution = 0.2
    end

    methods

        function obj = LaneDetector(varargin)
            % Construct an instance of this class
            parser = inputParser;
            parser.CaseSensitive = false;

            % Optional argument parsers
            addParameter(parser, 'ROI', obj.DefaultROI, @obj.validateROI);
            addParameter(parser, 'LaneWidth', obj.DefaultLaneWidth, @obj.validateLaneWidth);
            addParameter(parser, 'HistogramBinResolution', obj.DefaultHistogramBinResolution, @obj.validateHistogramBinResolution);
            addParameter(parser, 'VerticalBinResolution', obj.DefaultVerticalBinResolution, @obj.validateVerticalBinResolution);
            addParameter(parser, 'HorizontalBinResolution', obj.DefaultHorizontalBinResolution, @obj.validateHorizontalBinResolution);

            % Parse input arguments
            parse(parser, varargin{:});
            obj.ROI = parser.Results.ROI;
            obj.LaneWidth = parser.Results.LaneWidth;
            obj.VerticalBinResolution = parser.Results.VerticalBinResolution;
            obj.HorizontalBinResolution = parser.Results.HorizontalBinResolution;
            obj.HistogramBinResolution = parser.Results.HistogramBinResolution;
        end

        function lanePoints = detectLanes(obj, ptCloudIn)
            % Preprocess point cloud
            ptCloudIn = preprocessPointCloud(obj, ptCloudIn);

            % Extract lane start points using intensity histogram
            detection = computeStartPoint(obj, ptCloudIn);
            if ~detection
                lanePoints = [];
                return
            end
            % Extract lane points using window search
            lanePoints = detectLanesImpl(obj, ptCloudIn);

            % Refine lane points using fitting
            lanePoints = refineLanePoints(obj, lanePoints);
        end

        function lanes = updateLanePolynomial(obj, lanePolynomial)
            % 3D lane points are estimated using plane model and polynomial parameters.
            % Plane model is represented as ax + by + cz + d = 0
            % 2nd degree polynomial is represented as ax^2 + bx + c

            P1 = lanePolynomial(1, :);
            P2 = lanePolynomial(2, :);

            xval = linspace(obj.ROI(1), 40, 80);
            yval1 = polyval(P1, xval);
            yval2 = polyval(P2, xval);

            modelParams = obj.PlaneModel;
            zWorld1 = (-modelParams(1) * xval - modelParams(2) * yval1 - modelParams(4)) / modelParams(3);
            zWorld2 = (-modelParams(1) * xval - modelParams(2) * yval2 - modelParams(4)) / modelParams(3);

            lane3d1 = [xval', yval1', zWorld1'];
            lane3d2 = [xval', yval2', zWorld2'];
            lanes{1} = lane3d1;
            lanes{2} = lane3d2;
        end

    end

    methods (Access = private, Hidden)

        function [Ypk, Xpk] = findpeaks(~, histVal)
            % Finding peaks in the histogram
            x = (1:size(histVal, 2))';
            y = histVal';
            yTemp = [NaN; y; NaN];
            iTemp = (1:length(yTemp)).';

            % Keep only the first of any adjacent pairs of equal values (including NaN).
            yFinite = ~isnan(yTemp);
            iNeq = [1; 1 + find((yTemp(1:end - 1) ~= yTemp(2:end)) & (yFinite(1:end - 1) | yFinite(2:end)))];
            iTemp = iTemp(iNeq);

            % Take the sign of the first sample derivative
            s = sign(diff(yTemp(iTemp)));

            % Find local maxima
            iMax = 1 + find(diff(s) < 0);

            % Index into the original index vector without the NaN bookend.
            iPk = iTemp(iMax) - 1;

            % Fetch the coordinates of the peak
            Ypk = y(iPk)';
            Xpk = x(iPk)';
        end

        % Compute lane start point using intensity histogram
        function isLaneDetected = computeStartPoint(obj, ptCloud)
            [histVal, yvals] = computeHistogram(obj, ptCloud);
            [peaks, locs] = findpeaks(obj, histVal);
            startYs = yvals(locs);
            yvals1 = computeInitialWindow(obj, startYs, peaks);
            yvals2 = helperInitialWindow2(obj, startYs, peaks);
            isLaneDetected = false;
            if numel(yvals1) == 2 && numel(yvals2) == 2
                if abs((yvals1(2) - yvals1(1)) - obj.LaneWidth) > abs((yvals2(2) - yvals2(1)) - obj.LaneWidth)
                    yvals = yvals2;
                else
                    yvals = yvals1;
                end
                obj.StartLanePoints = yvals;
            end
            if ~isempty(yvals)
                isLaneDetected = true;
            end
        end

        % Intensity based histogram over ROI
        function [histVal, yvals] = computeHistogram(obj, ptCloud)
            numBins = ceil((ptCloud.YLimits(2) - ...
                            ptCloud.YLimits(1)) / obj.HistogramBinResolution);
            histVal = zeros(1, numBins - 1);
            binStartY = linspace(ptCloud.YLimits(1), ptCloud.YLimits(2), numBins);
            yvals = zeros(1, numBins - 1);
            for i = 1:numBins - 1
                roi = [-inf, 15, binStartY(i), binStartY(i + 1) -inf, inf];
                ind = findPointsInROI(ptCloud, roi);
                subPc = select(ptCloud, ind);
                if subPc.Count
                    histVal(i) = sum(subPc.Intensity);
                    yvals(i) = (binStartY(i) + binStartY(i + 1)) / 2;
                end
            end
        end

        % Finds the start window points using brute-force search
        function yval = computeInitialWindow(obj, yvals, peaks)
            leftLanesIndices = yvals >= 0;
            rightLanesIndices = yvals < 0;
            leftLaneYs = yvals(leftLanesIndices);
            rightLaneYs = yvals(rightLanesIndices);
            peaksLeft = peaks(leftLanesIndices);
            peaksRight = peaks(rightLanesIndices);
            diff = zeros(sum(leftLanesIndices), sum(rightLanesIndices));
            for i = 1:sum(leftLanesIndices)
                for j = 1:sum(rightLanesIndices)
                    diff(i, j) = abs(obj.LaneWidth - (leftLaneYs(i) - rightLaneYs(j)));
                end
            end
            [~, minIndex] = min(diff(:));
            [row, col] = ind2sub(size(diff), minIndex);
            yval = [leftLaneYs(row), rightLaneYs(col)];
            estimatedLaneWidth = leftLaneYs(row) - rightLaneYs(col);

            % If the calculated lane width is not within the bounds return
            % the lane with highest peak
            if abs(estimatedLaneWidth - obj.LaneWidth) > 0.5
                if peaksLeft(row) > peaksRight(col)
                    yval = leftLaneYs(row);
                else
                    yval = rightLaneYs(col);
                end
            end

            % Sort yval from min to max
            yval = sort(yval);
        end

        % Finds start window points using maximum peak as anchor point
        function startLanePoints = helperInitialWindow2(obj, startYs, peaks)
            [~, maxInd] = max(peaks);
            y1 = startYs(maxInd);

            % Find ys > y1
            ind1 = startYs > y1;
            ind2 = startYs < y1;
            dist1 = zeros(sum(ind1), 1);
            dist2 = zeros(sum(ind2), 1);
            startYs1 = startYs(ind1);
            startYs2 = startYs(ind2);
            for i = 1:sum(ind1)
                dist1(i, :) = abs(obj.LaneWidth - (startYs1(i) - y1));
            end
            for i = 1:sum(ind2)
                dist2(i, :) = abs(obj.LaneWidth - (y1 - startYs2(i)));
            end
            [~, minIndex1] = min(dist1);
            [~, minIndex2] = min(dist2);
            if isempty(minIndex1) || isempty(minIndex2)
                if isempty(minIndex1)
                    y2 = startYs2(minIndex2);
                else
                    y2 = startYs1(minIndex1);
                end
            else
                if min(dist1) < min(dist2)
                    y2 = startYs1(minIndex1);
                else
                    y2 = startYs2(minIndex2);
                end
            end
            if y2 > y1
                startLanePoints = [y1, y2];
            else
                startLanePoints = [y2, y1];
            end
        end

        % Extract lane points using window search
        function detectedLanePoints = detectLanesImpl(obj, ptCloud)
            numVerticalBins = ceil((ptCloud.XLimits(2) - ptCloud.XLimits(1)) / obj.VerticalBinResolution);
            laneStartX = linspace(ptCloud.XLimits(1), ptCloud.XLimits(2), numVerticalBins);
            numLanes = numel(obj.StartLanePoints);
            verticalBins = zeros(numVerticalBins, 3, numLanes);
            lanes = zeros(numVerticalBins, 3, numLanes);
            tmpStartY = obj.StartLanePoints;
            roi = repmat([-inf, inf], 1, 3); %#ok<NASGU>
            for i = 1:numVerticalBins - 1
                for j = 1:numLanes
                    laneStartY = tmpStartY(j);

                    % Define a vertical roi window
                    roi = [laneStartX(i), laneStartX(i + 1), laneStartY - obj.HorizontalBinResolution / 2, ...
                           laneStartY + obj.HorizontalBinResolution / 2, -inf, inf];
                    tmpPc = select(ptCloud, findPointsInROI(ptCloud, roi));
                    if ~isempty(tmpPc.Location)
                        [~, maxIndex] = max(tmpPc.Intensity);
                        val = tmpPc.Location(maxIndex, :);
                        verticalBins(i, :, j) = val;
                        lanes(i, :, j) = val;

                        % Slide the window with the update mean value along y direction
                        tmpStartY(j) = val(2);
                    else
                        value = lanes(2:end, 1:2, j);
                        value(all(value == 0, 2), :) = [];
                        if size(value, 1) == 2

                            % Use linear prediction
                            P = fitPolynomialRANSAC(value, 1, 0.1);
                        elseif size(value, 1) > 2

                            % Use 2 degree polynomial prediction
                            P = fitPolynomialRANSAC(value, 2, 0.1);
                        else
                            verticalBins(i, :, j) = verticalBins(end, :, j);
                            continue
                        end
                        error =  mean(sqrt((polyval(P, value(:, 1)) - value(:, 2)).^2));
                        if error < 0.1
                            xval = (roi(1) + roi(2)) / 2;
                            yval = polyval(P, xval);

                            % Use error to regularize the value of predicted y
                            yval = yval - error * abs(yval);
                            tmpStartY(j) = yval;
                            roi(3:4) = [yval - obj.HorizontalBinResolution, yval + obj.HorizontalBinResolution];

                            % Update the lane point with the centre of the predicted window
                            tmpPc = select(ptCloud, findPointsInROI(ptCloud, roi));
                            zmean = mean(tmpPc.Location(:, 3));
                            verticalBins(i, :, j) = [xval, yval, zmean];
                        else
                            roi(3:4) = [tmpStartY(j) - obj.HorizontalBinResolution, tmpStartY(j) + obj.HorizontalBinResolution]; %#ok<NASGU>
                        end

                    end
                end
            end
            if size(lanes, 3) > 1
                lane1 = lanes(:, :, 1);
                lane2 = lanes(:, :, 2);
                lane1(all(lane1 == 0, 2), :) = [];
                lane2(all(lane2 == 0, 2), :) = [];
                detectedLanePoints{1} = lane1;
                detectedLanePoints{2} = lane2;
                obj.LanesDetect = 2;
            else
                lane1 = lanes(:, :, 1);
                lane1(all(lane1 == 0, 2), :) = [];
                detectedLanePoints = lane1;
                obj.LanesDetect = 1;
            end
        end

        function refinedLanePoints = refineLanePoints(obj, lanePoints)
            if obj.LanesDetect > 1
                lane1 = lanePoints{1};
                lane2 = lanePoints{2};
                [P1, error1] = obj.fitPolynomial(lane1(:, 1:2), 2, 0.1);
                [P2, error2] = obj.fitPolynomial(lane2(:, 1:2), 2, 0.1);
                xval = linspace(obj.ROI(1), 40, 80);
                yval1 = polyval(P1, xval);
                yval2 = polyval(P2, xval);

                % Z coordinate estimation
                modelParams = obj.PlaneModel;
                zWorld1 = (-modelParams(1) * xval - modelParams(2) * yval1 - modelParams(4)) / modelParams(3);
                zWorld2 = (-modelParams(1) * xval - modelParams(2) * yval2 - modelParams(4)) / modelParams(3);
                lane3d1 = [xval', yval1', zWorld1'];
                lane3d2 = [xval', yval2', zWorld2'];

                % Shift the polynomial with high score along Y-axis towards the polynomial with low score
                if error1 > error2
                    lanePolynomial = P2;
                    if lane3d1(1, 2) > 0
                        lanePolynomial(3) = lane3d2(1, 2) + obj.LaneWidth;
                    else
                        lanePolynomial(3) = lane3d2(1, 2) - obj.LaneWidth;
                    end
                    lane3d1(:, 2) = polyval(lanePolynomial, lane3d1(:, 1));
                else
                    lanePolynomial = P1;
                    if lane3d2(1, 2) > 0
                        lanePolynomial(3) = lane3d1(1, 2) + obj.LaneWidth;
                    else
                        lanePolynomial(3) = lane3d1(1, 2) - obj.LaneWidth;
                    end
                    P2 = lanePolynomial;
                    lane3d2(:, 2) = polyval(lanePolynomial, lane3d2(:, 1));
                end
            else
                % For single lane
                lane1 = lanePoints;
                P1 = obj.fitPolynomial(lane1(:, 1:2), 2, 0.1);
                P2 = P1;
                xval = linspace(obj.ROI(1), 40, 80);
                yval1 = polyval(P1, xval);

                % Z coordinate estimation
                modelParams = obj.PlaneModel;
                zWorld1 = (-modelParams(1) * xval - modelParams(2) * yval1 - modelParams(4)) / modelParams(3);
                lane3d1 = [xval', yval1', zWorld1'];
                if lane3d1(1, 2) > 0
                    P2(3) = lane3d1(1, 2) - obj.LaneWidth;
                else
                    P2(3) = lane3d1(1, 2) + obj.LaneWidth;
                end
                yval2 = polyval(P2, xval);
                zWorld2 = (-modelParams(1) * xval - modelParams(2) * yval2 - modelParams(4)) / modelParams(3);
                lane3d2 = [xval', yval2', zWorld2'];

            end
            refinedLanePoints{1} = lane3d1;
            refinedLanePoints{2} = lane3d2;
            obj.LanePolynomial = [P1; P2];
        end

        % Ground extraction and outlier removal
        function [ground, obj] = preprocessPointCloud(obj, pc)
            if obj.isOrganised(pc)
                pc = removeInvalidPoints(pc);
            end
            ind = findPointsInROI(pc, obj.ROI);
            pc = select(pc, ind);

            % Remove Ego vehicle points
            nearIndices = findNeighborsInRadius(pc, [0 0 0], 2);
            nonEgoIndices = true(pc.Count, 1);
            nonEgoIndices(nearIndices) = false;
            indices = find(nonEgoIndices);
            pc = select(pc, indices);

            % Remove ground
            [model, inliers, outliers] = pcfitplane(pc, 0.1, [0, 0, 1]);
            obj.PlaneModel = model.Parameters;
            ground = select(pc, inliers);
            zlim = ground.ZLimits;
            nonGround = select(pc, outliers);
            [labels, numClusters] = pcsegdist(nonGround, 1);

            % Remove all the outlier from the ground occuring because of possible obstacles
            for i = 1:numClusters
                tmp = nonGround.Location(labels == i, :);
                xmin = min(tmp(:, 1));
                xmax = max(tmp(:, 1));
                ymin = min(tmp(:, 2));
                ymax = max(tmp(:, 2));
                roi = [xmin, xmax, ymin, ymax, zlim];
                ground = obj.cropPointCloudFromROI(ground, roi);
            end
        end

    end

    methods (Static, Access = private)

        function tf = validateROI(roi)
            if ~isempty(roi)
                if isnumeric(roi)
                    validateattributes(roi, {'single', 'double'}, {'real', 'nonsparse', 'nonnan', 'ncols', 6, 'nrows', 1}, mfilename, 'roi');
                    if any(roi(:, 1:2:5) > roi(:, 2:2:6))
                        error(message('lidar:lidarCameraCalibration:invalidROI'));
                    end
                else
                    error(message('lidar:lidarCameraCalibration:invalidROIDataType'));
                end
            end
            tf = true;
        end

        function [P, score] = fitPolynomial(pts, degree, resolution)
            P = fitPolynomialRANSAC(pts, degree, resolution);
            score =  mean(sqrt((polyval(P, pts(:, 1)) - pts(:, 2)).^2));
        end

        function ptCloud = cropPointCloudFromROI(ptCloud, roi)
            ind = findPointsInROI(ptCloud, roi);
            if ind
                flag = true(ptCloud.Count, 1);
                flag(ind) = false;
                outliers = find(flag);
                ptCloud = select(ptCloud, outliers);
            end
        end

        function tf = validateVerticalBinResolution(value)
            validateattributes(value, {'single', 'double'}, {'real', 'nonsparse', 'vector', 'nonnan', 'finite', 'numel', 1}, mfilename, 'verticalBinResolution');
            tf = true;
        end

        function tf = validateHorizontalBinResolution(value)
            validateattributes(value, {'single', 'double'}, {'real', 'nonsparse', 'vector', 'nonnan', 'finite', 'numel', 1}, mfilename, 'horizontalBinResolution');
            tf = true;
        end

        function tf = validateLaneWidth(value)
            validateattributes(value, {'single', 'double'}, {'real', 'nonsparse', 'vector', 'nonnan', 'finite', 'numel', 1}, mfilename, 'laneWidth');
            tf = true;
        end

        function tf = validateHistogramBinResolution(value)
            validateattributes(value, {'single', 'double'}, {'real', 'nonsparse', 'vector', 'nonnan', 'finite', 'numel', 1}, mfilename, 'histogramBinResolution');
            tf = true;
        end

        function tf = isOrganised(pc)
            if isequal(ndims(pc.Location), 3)
                tf = true;
            else
                tf = false;
            end
        end

    end
end
