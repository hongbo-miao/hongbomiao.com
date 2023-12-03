function T = readINSConfigFile(fileName)
    % Reads INS configuration file containing INS data, and returns it in a table.
    % This function expects data from the Velodyne SLAM Dataset.

    validateattributes(fileName, {'char', 'string'}, {'scalartext'}, mfilename, 'fileName');

    % Create options to read delimited text file
    opts = delimitedTextImportOptions;
    opts.Delimiter = ";";
    opts.DataLines = [8 inf];
    opts.VariableNames = ["Timestamps", "Num_Satellites", "Latitude", "Longitude", "Altitude", "Heading", "Pitch", "Roll", "Omega_Heading", "Omega_Pitch", "Omega_Roll", "V_X", "V_Y", "V_ZDown", "X", "Y", "Z"];
    opts.VariableTypes(2:end) = {'double'};

    T = readtable(fileName, opts);

    % Remove unnecessary column
    T.ExtraVar1 = [];

    % Convert timestamps to datetime
    T.Timestamps = datetime(T.Timestamps, 'InputFormat', 'yyyy-MM-dd HH:mm:ss.SSS');
    T = table2timetable(T);
end
