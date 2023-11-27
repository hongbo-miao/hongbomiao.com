function reflidarData = fetchLidarData()
    lidarDataTarFile = matlab.internal.examples.downloadSupportFile('lidar', 'data/WPI_LidarData.tar.gz');
    [outputFolder, ~, ~] = fileparts(lidarDataTarFile);

    % Check if tar.gz file is downloaded, but not uncompressed
    if ~exist(fullfile(outputFolder, 'WPI_LidarData.mat'), 'file')
        untar(lidarDataTarFile, outputFolder);
    end

    % Load lidar data
    load(fullfile(outputFolder, 'WPI_LidarData.mat'), 'lidarData');

    % Select region with a prominent intensity value
    reflidarData = cell(300, 1);
    count = 1;
    roi = [-50 50 -30 30 -inf inf];
    for i = 81:380
        pc = lidarData{i};
        ind = findPointsInROI(pc, roi);
        reflidarData{count} = select(pc, ind);
        count = count + 1;
    end
end
