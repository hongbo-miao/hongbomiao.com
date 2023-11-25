Sim_Time = 15;
F = 1;
m = 1;
b = 0.5;
k = 1;

sim('massSpringDamperSimscape.slx');
figure;
plot(ans.displacement);
hold on;
plot(ans.F);
ylabel('Displacement [m]');
xlabel('Time [sec]');
title('One Mass Spring Damper System Response');
