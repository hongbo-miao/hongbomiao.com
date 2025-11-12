function ptCloud = readPointCloudFromFile(fileName)
    % Reads point cloud data from the .png image file fileName and returns a pointCloud object.
    % Expects file to be from the Velodyne SLAM Dataset.

    % From DATAFORMAT.txt
    % -------------------
    % Each 360-degree revolution of the Velodyne scanner was stored as 16bit png distance image (scan*.png).
    % The scanner turned clockwise, filling the image from the leftmost to the rightmost column, with the leftmost and
    % rightmost column being at the back of the vehicle.
    % Note that measurements were not corrected for vehicle movement.
    % Thus and due to the physical setup of the laser diodes, some strange effects can be seen at the cut of the image when the vehicle is turning.
    % As consequence, it is best to ignore the 10 leftmost and rightmost columns of the image.
    % To convert the pixel values [0..65535] into meters, just divide by 500.
    % This results in an effective range of [0..131m]. Invalid measurements are indicated by zero distance.

    % To convert the distance values into 3D coordinates, use the setup in "img.cfg".
    % The yaw angles (counter-clockwise) are a linear mapping from the image column [0..869]->[180 to -180]
    % The pitch angles are specified for each image row separately.

    validateattributes(fileName, {'char', 'string'}, {'scalartext'}, mfilename, 'fileName');

    % Convert pixel values to range
    range = single(imread(fileName)) ./ 500;
    range(range == 0) = NaN;

    % Get yaw angles as a linear mapping of [0..869] -> [180 to -180].
    % These values are obtained from img.cfg file.
    yawAngles = 869:-1:0;
    yawAngles = -180 + yawAngles .* (360 / 869);
    pitchAngles = [-1.9367; -1.57397; -1.30476; -0.871566; -0.57881; -0.180617; 0.088762; 0.451829; 0.80315; 1.20124; 1.49388; 1.83324; 2.20757; 2.54663; 2.87384; 3.23588; 3.53933; 3.93585; 4.21552; 4.5881; 4.91379; 5.25078; 5.6106; 5.9584; 6.32889; 6.67575; 6.99904; 7.28731; 7.67877; 8.05803; 8.31047; 8.71141; 9.02602; 9.57351; 10.0625; 10.4707; 10.9569; 11.599; 12.115; 12.5621; 13.041; 13.4848; 14.0483; 14.5981; 15.1887; 15.6567; 16.1766; 16.554; 17.1868; 17.7304; 18.3234; 18.7971; 19.3202; 19.7364; 20.2226; 20.7877; 21.3181; 21.9355; 22.4376; 22.8566; 23.3224; 23.971; 24.5066; 24.9992];

    [yaw, pitch] = meshgrid(deg2rad(yawAngles), deg2rad(pitchAngles));
    rangeData = cat(3, range, pitch, yaw);

    xyzData = convertFromSphericalToCartesianCoordinates(rangeData);

    % Transform points so that coordinate system faces towards the front of the vehicle.
    ptCloud = pointCloud(xyzData .* cat(3, -1, 1, 1));
end

function xyzData = convertFromSphericalToCartesianCoordinates(rangeData)
    xyzData = zeros(size(rangeData), 'like', rangeData);

    range = rangeData(:, :, 1);
    pitch = rangeData(:, :, 2);
    yaw   = rangeData(:, :, 3);

    xyzData(:, :, 1) = range .* cos(pitch) .* sin(yaw);
    xyzData(:, :, 2) = range .* cos(pitch) .* cos(yaw);
    xyzData(:, :, 3) = -range .* sin(pitch);
end
