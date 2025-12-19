x = rand(1e9, 1, 'single');

% Automatically fused and GPU-accelerated
y = sin(x) .* x + 0.5;

% Result computed on GPU
disp(mean(y));
