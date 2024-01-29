% https://www.mathworks.com/help/aerotbx/ug/overlaying-simulated-and-actual-flight-data.html

h = Aero.Animation;

% timeStep = (1 / FramesPerSecond) * TimeScaling = 0.5s
h.FramesPerSecond = 10;
h.TimeScaling = 5;

% Simulated and actual flight trajectories
orangeAircraftPath = fullfile(matlabroot, 'toolbox', 'aero', 'astdemos', 'pa24-250_orange.ac');
blueAircraftPath = fullfile(matlabroot, 'toolbox', 'aero', 'astdemos', 'pa24-250_blue.ac');

idx1 = h.createBody(orangeAircraftPath, 'Ac3d');
idx2 = h.createBody(blueAircraftPath, 'Ac3d');

% "simdata" contains logged simulated data in a 6DoF array
simDataPath = fullfile(matlabroot, 'toolbox', 'aero', 'astdemos', 'simdata.mat');
load(simDataPath, 'simdata');
h.Bodies{1}.TimeSeriesSource = simdata;

% "fltdata" contains actual flight test data in a custom format
flightDataPath = fullfile(matlabroot, 'toolbox', 'aero', 'astdemos', 'fltdata.mat');
load(flightDataPath, 'fltdata');
h.Bodies{2}.TimeSeriesSource = fltdata;
h.Bodies{2}.TimeSeriesReadFcn = @readTimeSeriesData;
h.Bodies{2}.TimeSeriesSourceType = 'Custom';

% Camera
h.Camera.PositionFcn = @doFirstOrderChaseCameraDynamics;

h.play();
h.updateCamera(5);
h.wait();
h.delete();
