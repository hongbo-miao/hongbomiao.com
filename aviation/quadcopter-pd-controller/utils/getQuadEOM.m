function sdot = getQuadEOM(t, s, getControllerHandle, generateTrajectoryHandle, params)
    % Wrapper function for solving quadcopter equation of motion
    %   getQuadEOM takes in time, state vector, controller, trajectory generator
    %   and parameters and output the derivative of the state vector, the
    %   actual calcution is done in calculateQuadEOM.
    %
    % INPUTS:
    % t             - 1 x 1, time
    % s             - 13 x 1, state vector = [x, y, z, xd, yd, zd, qw, qx, qy, qz, p, q, r]
    % getControllerHandle - function handle of your controller
    % generateTrajectoryHandle    - function handle of your trajectory generator
    % params        - struct, output from getSysParams() and whatever parameters you want to pass in
    %
    % OUTPUTS:
    % sdot          - 13 x 1, derivative of state vector s
    %
    % See Also: calculateQuadEOM, getSysParams

    % convert state to quad stuct for control
    current_state = convertStateToQd(s);

    % Get desired_state
    desired_state = generateTrajectoryHandle(t, current_state);

    % get control outputs
    [F, M] = getControllerHandle(t, current_state, desired_state, params);

    % compute derivative
    sdot = calculateQuadEOM(t, s, F, M, params);
end
