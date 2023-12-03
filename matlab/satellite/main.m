% https://www.mathworks.com/help/satcom/gs/model-and-visualize-satelliteScenario.html

% Create a satellite scenario
startTime = datetime(2020, 1, 1, 0, 0, 0);
stopTime = startTime + hours(24);
sampleTimeS = 60;
scenario = satelliteScenario(startTime, stopTime, sampleTimeS);

% Add satellites with ground tracks
satellites = satellite(scenario, "threeSatelliteConstellation.tle");
leadTimeS = 1200;
trailTimeS = 1200;
groundTrack(satellites, "LeadTime", leadTimeS, "TrailTime", trailTimeS);

% Add ground stations
name = ["Madrid Deep Space Communications Complex", "Canberra Deep Space Communication Complex"];
lat = [40.42917, -35.40139];
lon = [-4.24917, 148.98167];
groundStations = groundStation(scenario, "Name", name, "Latitude", lat, "Longitude", lon);

play(scenario);

% Get satellites' orbital elements and positions
time = datetime(2020, 1, 1, 6, 0, 0);
orbitalElement1 = orbitalElements(satellites(1));
pos1 = states(satellites(1), time, "CoordinateFrame", "geographic");

% Get ground stations' azimuth angle, elevation angle, and eange
[azimuthAngle, elevationAngle, range] = aer(groundStations(1), satellites(1), time);
