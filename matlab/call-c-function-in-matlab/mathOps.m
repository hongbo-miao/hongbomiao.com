function [added, multed] = mathOps(n1, n2)
    added = 0;
    coder.cinclude('include/add.h');
    added = coder.ceval('add', n1, n2);

    multed = n1 * n2;
end
