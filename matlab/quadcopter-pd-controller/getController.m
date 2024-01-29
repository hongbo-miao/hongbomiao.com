function [F, M] = getController(t, state, des_state, params)
    % Controller for the quadcopter
    %
    %   state: The current state of the robot with the following fields:
    %   state.pos = [x; y; z], state.vel = [x_dot; y_dot; z_dot],
    %   state.rot = [phi; theta; psi], state.omega = [p; q; r]
    %
    %   des_state: The desired states are:
    %   des_state.pos = [x; y; z], des_state.vel = [x_dot; y_dot; z_dot],
    %   des_state.acc = [x_ddot; y_ddot; z_ddot], des_state.yaw,
    %   des_state.yawdot
    %
    %   params: robot parameters

    % Gains
    Kd = [1; 1; 1];
    Kp = [100; 100; 800];
    Kd_ang = [1; 1; 1];
    Kp_ang = [160; 160; 160];

    grav = params.gravity;
    mass = params.mass;

    % Compute command accelerations
    cmd_accel = des_state.acc + Kd .* (des_state.vel - state.vel) + Kp .* (des_state.pos - state.pos);

    % Thrust
    F = mass * (grav + cmd_accel(3));
    F = min(max(F, params.minF), params.maxF);

    % Desired roll and pitch
    phi_des = (1 / grav) * (cmd_accel(1) * sin(des_state.yaw) - cmd_accel(2) * cos(des_state.yaw));
    theta_des = (1 / grav) * (cmd_accel(1) * cos(des_state.yaw) + cmd_accel(2) * sin(des_state.yaw));

    % Desired angle vector and angle rate vector
    rot_des = [phi_des; theta_des; des_state.yaw];
    omega_des = [0; 0; des_state.yawdot];

    % Moment
    M = Kp_ang .* (rot_des - state.rot) + Kd_ang .* (omega_des - state.omega);
end
