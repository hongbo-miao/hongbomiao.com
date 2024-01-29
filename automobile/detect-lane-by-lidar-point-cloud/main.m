% https://www.mathworks.com/help/lidar/ug/lane-detection-in-3d-lidar-point-cloud.html

% Download lidar data
lidarData = fetchLidarData();

% Select first frame
ptCloud = lidarData{1};

% Define ROI in meters
xlim = [5 55];
ylim = [-3 3];
zlim = [-4 1];
roi = [xlim ylim zlim];

% Crop point cloud using ROI
indices = findPointsInROI(ptCloud, roi);
croppedPtCloud = select(ptCloud, indices);

% Remove ground plane
maxDistance = 0.1;
referenceVector = [0 0 1];
[model, inlierIndices, outlierIndices] = pcfitplane(croppedPtCloud, maxDistance, referenceVector);
groundPts = select(croppedPtCloud, inlierIndices);

% Peak Intensity Detection
histBinResolution = 0.2;
[histVal, yvals] = computeHistogram(groundPts, histBinResolution);

% Obtain peaks in the histogram by using the helperfindpeaks helper function.
% Further filter the possible lane points based on lane width using initializeWindow.
[peaks, locs] = findPeaks(histVal);
startYs = yvals(locs);

laneWidth = 4;
[startLanePoints, detectedPeaks] = initializeWindow(startYs, peaks, laneWidth);

% Window Initialization
vBinRes = 1.0;
hBinRes = 0.8;
numVerticalBins = ceil((groundPts.XLimits(2) - groundPts.XLimits(1)) / vBinRes);

% Sliding Window
lanes = detectLanes(groundPts, hBinRes, numVerticalBins, startLanePoints);

% Plot final lane points
lane1 = lanes{1};
lane2 = lanes{2};

% Lane Fitting
% Parabolic Polynomial Fitting
[P1, error1] = fitPolynomial(lane1(:, 1:2), 2, 0.1);
[P2, error2] = fitPolynomial(lane2(:, 1:2), 2, 0.1);

xval = linspace(5, 40, 80);
yval1 = polyval(P1, xval);
yval2 = polyval(P2, xval);

% Z-coordinate estimation
modelParams = model.Parameters;
zWorld1 = (-modelParams(1) * xval - modelParams(2) * yval1 - modelParams(4)) / modelParams(3);
zWorld2 = (-modelParams(1) * xval - modelParams(2) * yval2 - modelParams(4)) / modelParams(3);

% Parallel Lane Fitting
lane3d1 = [xval' yval1' zWorld1'];
lane3d2 = [xval' yval2' zWorld2'];

% Shift the polynomial with a high score along the Y-axis towards
% the polynomial with a low score
if error1 > error2
    lanePolynomial = P2;
    if lane3d1(1, 2) > 0
        lanePolynomial(3) = lane3d2(1, 2) + laneWidth;
    else
        lanePolynomial(3) = lane3d2(1, 2) - laneWidth;
    end
    lane3d1(:, 2) = polyval(lanePolynomial, lane3d1(:, 1));
    lanePolynomials = [lanePolynomial; P2];
else
    lanePolynomial = P1;
    if lane3d2(1, 2) > 0
        lanePolynomial(3) = lane3d1(1, 2) + laneWidth;
    else
        lanePolynomial(3) = lane3d1(1, 2) - laneWidth;
    end
    lane3d2(:, 2) = polyval(lanePolynomial, lane3d2(:, 1));
    lanePolynomials = [P1; lanePolynomial];
end

% Lane Tracking
% Initial values
curveInitialParameters = lanePolynomials(1, 1:2);
driftInitialParameters = lanePolynomials(:, 3)';
initialEstimateError = [1 1 1] * 1e-1;
motionNoise = [1 1 1] * 1e-7;
measurementNoise = 10;

% Configure Kalman filter
curveFilter = configureKalmanFilter('ConstantAcceleration', curveInitialParameters, initialEstimateError, motionNoise, measurementNoise);
driftFilter = configureKalmanFilter('ConstantAcceleration', driftInitialParameters, initialEstimateError, motionNoise, measurementNoise);

% Loop Through Data
% Initialize the random number generator
rng(2020);
numFrames = numel(lidarData);
detector = LaneDetector('ROI', [5 40 -3 3 -4 1]);

% Turn on display
player = pcplayer([0 50], [-15 15], [-5 5]);

drift = zeros(numFrames, 1);
filteredDrift = zeros(numFrames, 1);
curveSmoothness = zeros(numFrames, 1);
filteredCurveSmoothness = zeros(numFrames, 1);
for i = 1:numFrames
    ptCloud = lidarData{i};

    % Detect lanes
    detectLanes(detector, ptCloud);

    % Predict polynomial from Kalman filter
    predict(curveFilter);
    predict(driftFilter);

    % Correct polynomial using Kalman filter
    lanePolynomials = detector.LanePolynomial;
    drift(i) = mean(lanePolynomials(:, 3));
    curveSmoothness(i) = mean(lanePolynomials(:, 1));
    updatedCurveParams = correct(curveFilter, lanePolynomials(1, 1:2));
    updatedDriftParams = correct(driftFilter, lanePolynomials(:, 3)');

    % Update lane polynomials
    updatedLanePolynomials = [repmat(updatedCurveParams, [2 1]), updatedDriftParams'];

    % Estimate new lane points with updated polynomial
    lanes = updateLanePolynomial(detector, updatedLanePolynomials);
    filteredDrift(i) = mean(updatedLanePolynomials(:, 3));
    filteredCurveSmoothness(i) = mean(updatedLanePolynomials(:, 1));

    % Visualize lanes after parallel fitting
    ptCloud.Color = uint8(repmat([0 0 255], ptCloud.Count, 1));
    lane3dPc1 = pointCloud(lanes{1});
    lane3dPc1.Color = uint8(repmat([255 0 0], lane3dPc1.Count, 1));
    lanePc = pccat([ptCloud lane3dPc1]);
    lane3dPc2 = pointCloud(lanes{2});
    lane3dPc2.Color = uint8(repmat([255 255 0], lane3dPc2.Count, 1));
    lanePc = pccat([lanePc lane3dPc2]);
    view(player, lanePc);
end
