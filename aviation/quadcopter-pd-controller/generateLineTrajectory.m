function [desired_state] = generateLineTrajectory(t, ~)
    t_max = 4;
    t = max(0, min(t, t_max));
    t = t / t_max;

    pos = 10 * t.^3 - 15 * t.^4 + 6 * t.^5;
    vel = (30 / t_max) * t.^2 - (60 / t_max) * t.^3 + (30 / t_max) * t.^4;
    acc = (60 / t_max^2) * t - (180 / t_max^2) * t.^2 + (120 / t_max^2) * t.^3;

    % output desired state
    desired_state.pos = [pos; pos; pos];
    desired_state.vel = [vel; vel; vel];
    desired_state.acc = [acc; acc; acc];
    desired_state.yaw = pos;
    desired_state.yawdot = vel;
end
