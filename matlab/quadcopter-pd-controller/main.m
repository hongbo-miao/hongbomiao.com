%% Trajectory
% generateTrajectoryHandle = @generateLineTrajectory;
generateTrajectoryHandle = @generateHelixTrajectory;

%% Controller
getControllerHandle = @getController;

% Run simulation with given trajectory generator and controller
% state - n x 13, with each row having format [x, y, z, xdot, ydot, zdot, qw, qx, qy, qz, p, q, r]
[t, state] = simulate(generateTrajectoryHandle, getControllerHandle);
